import torch
from utils import MockPassageEncoder, MockKnowledgeBase, save_embeddings
import os
import time

def re_encoding_process(manager, device_id=0, embedding_dim=768, batch_size=512):
    """Asynchronously re-encodes the knowledge base using the latest passage encoder."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu")
    print(f"[Re-encoding Process] Running on device: {device}")

    passage_encoder = MockPassageEncoder(embedding_dim=embedding_dim).to(device)
    knowledge_base = MockKnowledgeBase(num_passages=10000) # Smaller for simulation

    while True:
        # Wait for a signal to update the encoder state
        if manager.get_encoder_update_flag():
            print("[Re-encoding Process] Received signal to update encoder state.")
            encoder_state = manager.get_passage_encoder_state()
            if encoder_state:
                # Convert dict back to proper state_dict (tensors)
                loaded_state_dict = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in encoder_state.items()}
                passage_encoder.load_state_dict(loaded_state_dict)
                print("[Re-encoding Process] Passage encoder state updated.")
                manager.set_encoder_update_flag(False)

                print("[Re-encoding Process] Starting knowledge base re-encoding...")
                all_embeddings = []
                start_time = time.time()
                for i, batch in enumerate(knowledge_base.get_passages_batch(batch_size=batch_size)):
                    with torch.no_grad():
                        embeddings = passage_encoder(batch.to(device)).cpu()
                        all_embeddings.append(embeddings)
                    if (i + 1) % 10 == 0:
                        print(f"[Re-encoding Process] Processed {len(all_embeddings) * batch_size} passages.")
                
                updated_embeddings = torch.cat(all_embeddings, dim=0)
                embedding_save_path = f"./data/embeddings/medical_embeddings_{int(time.time())}.pt"
                save_embeddings(updated_embeddings, embedding_save_path)
                manager.set_re_encoded_embeddings_path(embedding_save_path) # Signal to re-indexing
                print(f"[Re-encoding Process] Re-encoding complete in {time.time() - start_time:.2f} seconds.")
            else:
                print("[Re-encoding Process] Encoder state was empty, skipping update.")

        time.sleep(5) # Check for updates every 5 seconds

if __name__ == '__main__':
    from multiprocessing import Process
    from knowledge_base_manager import KnowledgeBaseManager

    # This block is for testing the re-encoding process independently
    manager = KnowledgeBaseManager()
    manager.set_re_encoded_embeddings_path("") # Initialize
    manager.set_updated_index_ready(False) # Initialize

    p = Process(target=re_encoding_process, args=(manager, -1)) # Use CPU for independent test
    p.start()

    # Simulate updating the passage encoder state from a main training loop
    mock_encoder = MockPassageEncoder()
    mock_state = mock_encoder.get_state_dict()
    # Convert tensors to lists for Manager.dict compatibility
    serializable_state = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in mock_state.items()}
    manager.set_passage_encoder_state(serializable_state)

    # Let it run for a bit
    time.sleep(20)
    
    print("[Re-encoding Process Test] Latest embeddings path:", manager.get_re_encoded_embeddings_path())
    p.terminate()
    p.join()
    manager.terminate()
