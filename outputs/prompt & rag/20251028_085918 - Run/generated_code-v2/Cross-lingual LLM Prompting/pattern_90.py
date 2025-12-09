class SimulatedEmbeddingModel:
    def get_embedding(self, text):
        # In a real application, this would use a sentence-transformer model
        # For simulation, return a fixed-size list of floats based on the hash of the text
        import hashlib
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (10**8)
        # Using a simple deterministic 