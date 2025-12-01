import concurrent.futures
import time

def process_image(product_data):
    time.sleep(1) # Simulate image processing time
    return {"image_status": "processed", "image_url": f"https://example.com/images/{product_data['product_id']}_processed.jpg"}

def generate_description(product_data):
    time.sleep(2) # Simulate LLM description generation time
    return {"description": f"A detailed description for {product_data['name']} with ID {product_data['product_id']}. This is a fantastic product!"}

def classify_category(product_data):
    time.sleep(0.5) # Simulate category classification time
    return {"category": "Electronics" if "phone" in product_data['name'].lower() else "General Merchandise"}

def generate_seo_keywords(product_data):
    time.sleep(1.5) # Simulate SEO keyword generation time
    return {"seo_keywords": [f"{product_data['name']} review", f"buy {product_data['name']}", "best product"]}

def validate_price(product_data):
    time.sleep(0.3) # Simulate price validation time
    is_valid = product_data['price'] > 0
    return {"price_valid": is_valid, "validation_message": "Price is valid" if is_valid else "Price must be greater than 0"}

def update_inventory(product_data):
    time.sleep(0.8) # Simulate inventory system update time
    return {"inventory_status": "updated", "stock_level": product_data.get('stock', 100)}

def update_main_database(product_id, processed_data):
    time.sleep(2.5) # Simulate database commit time
    return {"database_status": "success", "product_id": product_id, "final_data": processed_data}


class ProductCatalogManager:
    def __init__(self, max_workers=5):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def process_product_update(self, product_data):
        print(f"\nStarting processing for product: {product_data['name']} (ID: {product_data['product_id']})")
        start_time = time.time()

        futures = {}
        # Independent tasks
        futures['image_processing'] = self.executor.submit(process_image, product_data)
        futures['description_generation'] = self.executor.submit(generate_description, product_data)
        futures['category_classification'] = self.executor.submit(classify_category, product_data)
        futures['seo_keyword_generation'] = self.executor.submit(generate_seo_keywords, product_data)
        futures['price_validation'] = self.executor.submit(validate_price, product_data)
        futures['inventory_update'] = self.executor.submit(update_inventory, product_data)

        processed_results = {}
        for task_name, future in futures.items():
            try:
                result = future.result()
                processed_results[task_name] = result
                print(f"  Task '{task_name}' completed: {result}")
            except Exception as exc:
                print(f"  Task '{task_name}' generated an exception: {exc}")
                processed_results[task_name] = {"error": str(exc)}
        
        # Dependent task: Update main database after all independent tasks are done
        print("All independent tasks completed. Proceeding with database update.")
        database_future = self.executor.submit(update_main_database, product_data['product_id'], processed_results)
        try:
            db_result = database_future.result()
            processed_results['database_update'] = db_result
            print(f"  Task 'database_update' completed: {db_result}")
        except Exception as exc:
            print(f"  Task 'database_update' generated an exception: {exc}")
            processed_results['database_update'] = {"error": str(exc)}

        end_time = time.time()
        print(f"Finished processing for product {product_data['product_id']} in {end_time - start_time:.2f} seconds.")
        return processed_results

    def shutdown(self):
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    manager = ProductCatalogManager(max_workers=5)

    product1 = {
        "product_id": "P001",
        "name": "Smartphone X",
        "price": 799.99,
        "description_draft": "A powerful new smartphone.",
        "stock": 150
    }

    product2 = {
        "product_id": "P002",
        "name": "Wireless Earbuds Z",
        "price": 129.00,
        "description_draft": "High-fidelity wireless sound.",
        "stock": 300
    }

    product3 = {
        "product_id": "P003",
        "name": "Empty Product",
        "price": 0,
        "description_draft": "",
        "stock": 50
    }

    # Process products sequentially to demonstrate the manager's functionality
    # The internal tasks for each product will run in parallel
    results1 = manager.process_product_update(product1)
    print("\n--- Consolidated Results for Product P001 ---")
    for k, v in results1.items():
        print(f"{k}: {v}")

    results2 = manager.process_product_update(product2)
    print("\n--- Consolidated Results for Product P002 ---")
    for k, v in results2.items():
        print(f"{k}: {v}")

    results3 = manager.process_product_update(product3)
    print("\n--- Consolidated Results for Product P003 ---")
    for k, v in results3.items():
        print(f"{k}: {v}")

    manager.shutdown()
    print("\nProductCatalogManager shut down.")