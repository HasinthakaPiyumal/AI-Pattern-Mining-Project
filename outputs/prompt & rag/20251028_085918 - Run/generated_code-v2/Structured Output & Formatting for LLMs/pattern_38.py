import random

def extract_unstructured_data(product_id: str) -> str:
    """
    Simulates the extraction of unstructured product data from various sources.
    In a real-world scenario, this would involve parsing PDFs, scraping web pages,
    or reading from vendor feeds.

    Args:
        product_id: A unique identifier for the product.

    Returns:
        A string containing simulated unstructured product information.
    """
    sample_data = {
        "P001": (
            "Product Name: Super Comfort Ergonomic Office Chair\n" +
            "Description: This chair offers unparalleled comfort and support " +
            "for long working hours. Features include adjustable lumbar support, " +
            "90-135 degree recline, breathable mesh back, and a sturdy five-star base. " +
            "Material: High-quality mesh, PU leather, steel frame. " +
            "Price: $299.99. Category: Office Furniture. Weight: 18kg. " +
            "Colors available: Black, Grey. Manufactured by: ErgoSeating Co."
        ),
        "P002": (
            "GizmoTech Smartwatch X: Track your fitness, monitor heart rate, " +
            "receive notifications, and more! Compatible with iOS and Android. " +
            "Long-lasting battery. Water-resistant. Price $149.00. " +
            "Category: Electronics. Display: 1.5-inch AMOLED. " +
            "Special features: GPS, Sleep Monitor, Calorie Counter. Brand: GizmoTech."
        ),
        "P003": (
            "The Ultimate Coffee Maker 2.0. Brew perfect coffee every time. " +
            "Capacity: 12 cups. Features: Programmable timer, auto-shutoff, " +
            "keep-warm function. Material: Stainless Steel and BPA-free plastic. " +
            "Price: 75.50 USD. Category: Kitchen Appliances. " +
            "Includes reusable filter. Model No.: CM2000. Voltage: 120V."
        )
    }

    return sample_data.get(product_id, (
        "Generic Product Data: This is some unstructured text about a product. " +
        "It might contain details like name, description, price, features, and brand. " +
        "The exact format varies wildly, requiring AI for standardization. " +
        "Example feature: Eco-friendly materials. Example price: $X.XX. " +
        "Category might be Home Goods. Brand: Unknown. Model: XYZ." ))