import multiprocessing
import time
import os
import shutil
import numpy as np
import faiss

# --- Constants and Shared Paths ---
SHARED_DIR = "./rag_shared_data"
ENCODER_WEIGHTS_PREFIX = "encoder_weights_"
FAISS_INDEX_PREFIX = "faiss_index_"
DUMMY_EMBEDDING_DIM = 768

# --- Mock Components ---
class MockPassageEncoder:
    def __init__(self, model_id):
        self.model_id = model_id
        self.weights_version = 0

    def encode(self, texts):
        # Simulate embedding generation
        time.sleep(0.1)  # Simulate computation time
        return np.random.rand(len(texts), DUMMY_EMBEDDING_DIM).astype('float32')

    def save_weights(self, path):
        with open(path, "w") as f:
            f.write(f"MockEncoderWeights_v{self.weights_version}_id{self.model_id}")
        self.weights_version += 1

    def load_weights(self, path):
        # In a real scenario, load PyTorch/TF state_dict
        with open(path, "r") as f:
            _ = f.read()
        # Simulate loading, update internal state if necessary
        self.weights_version = int(path.split('_v')[-1].split('_')[0])

class MockRAGModel:
    def __init__(self, initial_faiss_index_path, embedding_dim):
        self.current_faiss_index = None
        self.embedding_dim = embedding_dim
        self.load_faiss_index(initial_faiss_index_path)
        self.encoder = MockPassageEncoder("rag_main_encoder")
        self.training_step = 0

    def load_faiss_index(self, path):
        if os.path.exists(path):
            self.current_faiss_index = faiss.read_index(path)
            # print(f"Main RAG Loop: Loaded new FAISS index from {path}")
        else:
            # print(f"Main RAG Loop: Initializing dummy FAISS index as {path} not found")
            self.current_faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def train_step(self, query, ground_truth_answer):
        # Simulate RAG training step: retrieval, generation, loss calculation, gradient update
        time.sleep(0.05) # Simulate training computation
        self.training_step += 1
        # print(f"Main RAG Loop: Training step {self.training_step}")

    def get_current_encoder_weights_path(self, version):
        filename = f"{ENCODER_WEIGHTS_PREFIX}{version}.pt"
        return os.path.join(SHARED_DIR, filename)


# --- Helper Functions ---
def _generate_dummy_documents(start_id, num_docs):
    documents = []
    for i in range(num_docs):
        doc_id = f"doc_{start_id + i}"
        content = f"This is medical research document {doc_id} with some relevant text."
        documents.append((doc_id, content))
    return documents


def _save_faiss_index(index, path):
    faiss.write_index(index, path)


def _load_faiss_index(path):
    return faiss.read_index(path)


# --- Processes ---
def re_encoding_process(encoder_update_q, documents_to_encode_q, embeddings_to_index_q, shared_dir):
    current_encoder = MockPassageEncoder("re_encoder_process")
    print("Re-encoding Process: Started")
    while True:
        try:
            # Check for updated encoder weights
            if not encoder_update_q.empty():
                encoder_weights_path = encoder_update_q.get()
                current_encoder.load_weights(encoder_weights_path)
                print(f"Re-encoding Process: Loaded new encoder weights from {encoder_weights_path}")

            # Check for documents to encode
            if not documents_to_encode_q.empty():
                doc_batch = documents_to_encode_q.get()
                doc_ids = [doc[0] for doc in doc_batch]
                doc_contents = [doc[1] for doc in doc_batch]

                embeddings = current_encoder.encode(doc_contents)
                
                for i, doc_id in enumerate(doc_ids):
                    embeddings_to_index_q.put((doc_id, embeddings[i]))
                print(f"Re-encoding Process: Encoded {len(doc_batch)} documents.")

            time.sleep(0.5) # Prevent busy-waiting
        except KeyboardInterrupt:
            break
    print("Re-encoding Process: Exiting")

def re_indexing_process(embeddings_to_index_q, new_index_ready_event, new_index_path_q, shared_dir, embedding_dim):
    print("Re-indexing Process: Started")
    current_embeddings_data = [] # List of (doc_id, embedding)
    index_version = 0
    
    while True:
        try:
            # Collect embeddings
            while not embeddings_to_index_q.empty():
                item = embeddings_to_index_q.get()
                current_embeddings_data.append(item)

            # Periodically rebuild/update index if enough new embeddings
            if len(current_embeddings_data) > 10: # Threshold for rebuilding
                print(f"Re-indexing Process: Building new index with {len(current_embeddings_data)} embeddings.")
                
                # Extract embeddings and IDs
                doc_ids = [item[0] for item in current_embeddings_data]
                embeddings = np.array([item[1] for item in current_embeddings_data])
                
                # Create a new FAISS index (for simplicity, we rebuild from scratch)
                new_index = faiss.IndexFlatL2(embedding_dim)
                new_index.add(embeddings)
                
                index_version += 1
                new_index_filename = f"{FAISS_INDEX_PREFIX}{index_version}.faiss"
                new_index_path = os.path.join(shared_dir, new_index_filename)
                _save_faiss_index(new_index, new_index_path)
                
                # Signal main loop
                new_index_path_q.put(new_index_path)
                new_index_ready_event.set()
                print(f"Re-indexing Process: New FAISS index saved to {new_index_path} and signaled main loop.")
                current_embeddings_data = [] # Clear collected data after indexing

            time.sleep(1) # Prevent busy-waiting
        except KeyboardInterrupt:
            break
    print("Re-indexing Process: Exiting")

def main_rag_training_loop(encoder_update_q, documents_to_encode_q, new_index_ready_event, new_index_path_q, shared_dir, embedding_dim):
    print("Main RAG Training Loop: Started")
    
    # Initialize dummy FAISS index and RAG model
    initial_index_path = os.path.join(shared_dir, f"{FAISS_INDEX_PREFIX}0.faiss")
    if not os.path.exists(initial_index_path):
        dummy_embeddings = np.random.rand(50, embedding_dim).astype('float32')
        initial_index = faiss.IndexFlatL2(embedding_dim)
        initial_index.add(dummy_embeddings)
        _save_faiss_index(initial_index, initial_index_path)

    rag_model = MockRAGModel(initial_index_path, embedding_dim)

    encoder_version_counter = 0
    document_id_counter = 0
    
    for epoch in range(50):
        # Simulate a training epoch
        for step in range(10):
            rag_model.train_step(query="medical question", ground_truth_answer="relevant answer")

            # Periodically update encoder weights and send new documents
            if rag_model.training_step % 5 == 0:
                # Save current encoder weights and send path to re-encoding process
                encoder_version_counter += 1
                encoder_weights_path = rag_model.get_current_encoder_weights_path(encoder_version_counter)
                rag_model.encoder.save_weights(encoder_weights_path)
                encoder_update_q.put(encoder_weights_path)
                print(f"Main RAG Loop: Sent encoder weights v{encoder_version_counter} to re-encoding process.")

                # Simulate new documents arriving and send to re-encoding process
                new_docs = _generate_dummy_documents(document_id_counter, 3)
                documents_to_encode_q.put(new_docs)
                document_id_counter += 3
                print(f"Main RAG Loop: Sent {len(new_docs)} new documents for re-encoding.")

            # Check for new FAISS index from re-indexing process
            if new_index_ready_event.is_set():
                new_index_path = new_index_path_q.get()
                rag_model.load_faiss_index(new_index_path)
                new_index_ready_event.clear()
                print(f"Main RAG Loop: Switched to new FAISS index: {new_index_path}")

        time.sleep(0.2) # Simulate time between epochs
    
    print("Main RAG Training Loop: Finished. Signaling child processes to exit.")
    # Use a sentinel value or a specific message to signal processes to exit gracefully in a real app


if __name__ == "__main__":
    # Clean up previous run's data
    if os.path.exists(SHARED_DIR):
        shutil.rmtree(SHARED_DIR)
    os.makedirs(SHARED_DIR)

    # Initialize communication queues and events
    encoder_update_queue = multiprocessing.Queue()
    documents_to_encode_queue = multiprocessing.Queue()
    embeddings_to_index_queue = multiprocessing.Queue()
    new_index_ready_event = multiprocessing.Event()
    new_index_path_queue = multiprocessing.Queue() # To pass the path of the new index

    # Create and start processes
    re_encoder_p = multiprocessing.Process(
        target=re_encoding_process,
        args=(encoder_update_queue, documents_to_encode_queue, embeddings_to_index_queue, SHARED_DIR)
    )
    re_indexer_p = multiprocessing.Process(
        target=re_indexing_process,
        args=(embeddings_to_index_queue, new_index_ready_event, new_index_path_queue, SHARED_DIR, DUMMY_EMBEDDING_DIM)
    )
    main_loop_p = multiprocessing.Process(
        target=main_rag_training_loop,
        args=(encoder_update_queue, documents_to_encode_queue, new_index_ready_event, new_index_path_queue, SHARED_DIR, DUMMY_EMBEDDING_DIM)
    )

    re_encoder_p.start()
    re_indexer_p.start()
    main_loop_p.start()

    # Wait for processes to complete (or for a KeyboardInterrupt)
    try:
        main_loop_p.join() # Wait for the main loop to finish its simulated training
    except KeyboardInterrupt:
        print("\nMain process received KeyboardInterrupt. Terminating children.")
    finally:
        # Terminate child processes
        if re_encoder_p.is_alive():
            re_encoder_p.terminate()
            re_encoder_p.join()
        if re_indexer_p.is_alive():
            re_indexer_p.terminate()
            re_indexer_p.join()
        
        print("All processes terminated.")
        # Clean up shared directory
        if os.path.exists(SHARED_DIR):
            shutil.rmtree(SHARED_DIR)
        print("Cleaned up shared directory.")
