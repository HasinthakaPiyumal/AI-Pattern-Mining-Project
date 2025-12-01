class DataLoader:
    """Loads a simulated catalog of movies and TV shows."""
    def __init__(self):
        self.catalog = self._load_mock_catalog()

    def _load_mock_catalog(self):
        """Simulates loading a movie/TV show catalog."""
        return [
            {"id": 1, "title": "The Matrix", "genre": "Sci-Fi", "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.", "year": 1999},
            {"id": 2, "title": "Inception", "genre": "Sci-Fi, Action", "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.", "year": 2010},
            {"id": 3, "title": "Pulp Fiction", "genre": "Crime, Drama", "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.", "year": 1994},
            {"id": 4, "title": "The Dark Knight", "genre": "Action, Crime, Drama", "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.", "year": 2008},
            {"id": 5, "title": "Interstellar", "genre": "Sci-Fi, Drama", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "year": 2014},
            {"id": 6, "title": "Friends", "genre": "Comedy, Romance", "description": "Follows the personal and professional lives of six twenty-somethings living in Manhattan.", "year": 1994},
            {"id": 7, "title": "Stranger Things", "genre": "Sci-Fi, Horror", "description": "When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces, and one strange little girl.", "year": 2016},
            {"id": 8, "title": "The Crown", "genre": "Biography, Drama, History", "description": "Follows the political rivalries and romance of Queen Elizabeth II's reign and the events that shaped the second half of the 20th century.", "year": 2016},
            {"id": 9, "title": "The Office (US)", "genre": "Comedy", "description": "A mockumentary on the everyday lives of a group of office workers in the Scranton, Pennsylvania, branch of the fictional Dunder Mifflin Paper Company.", "year": 2005},
            {"id": 10, "title": "Breaking Bad", "genre": "Crime, Drama, Thriller", "description": "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future.", "year": 2008}
        ]

    def get_catalog(self):
        """Returns the loaded catalog."""
        return self.catalog