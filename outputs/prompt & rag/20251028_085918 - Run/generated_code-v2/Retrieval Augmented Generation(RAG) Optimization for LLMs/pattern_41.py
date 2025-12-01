
class KnowledgeBase:
    def __init__(self):
        self.documents = [
            {
                "id": "doc_1",
                "content": "Our return policy states that items can be returned within 30 days of purchase, provided they are in original condition with a receipt."
            },
            {
                "id": "doc_2",
                "content": "Shipping usually takes 3-5 business days for standard delivery within the country. Express shipping options are available at checkout."
            },
            {
                "id": "doc_3",
                "content": "To reset your password, please visit our website, click 'Login', then 'Forgot Password', and follow the instructions sent to your email."
            },
            {
                "id": "doc_4",
                "content": "Our customer support team is available Monday to Friday, 9 AM to 5 PM EST. You can reach us via phone, email, or live chat."
            },
            {
                "id": "doc_5",
                "content": "We accept all major credit cards, PayPal, and Apple Pay. Unfortunately, we do not accept personal checks."
            },
            {
                "id": "doc_6",
                "content": "Product warranties vary by item. Please check the product description page for specific warranty information. Most electronics come with a 1-year warranty."
            },
            {
                "id": "doc_7",
                "content": "You can track your order by entering your tracking number on our 'Order Tracking' page, which is accessible from the main menu."
            },
            {
                "id": "doc_8",
                "content": "For bulk orders, please contact our sales department directly to discuss special pricing and delivery options."
            },
            {
                "id": "doc_9",
                "content": "Our privacy policy details how we collect, use, and protect your personal data. It is fully compliant with GDPR and CCPA regulations."
            },
            {
                "id": "doc_10",
                "content": "We offer gift wrapping services for a small additional fee. You can select this option during the checkout process."
            }
        ]

    def get_document_by_id(self, doc_id):
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None

    def get_all_documents(self):
        return self.documents
