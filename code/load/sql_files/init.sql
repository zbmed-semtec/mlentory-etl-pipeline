USE Extraction_Results;

CREATE TABLE IF NOT EXISTS triples (
        id INT AUTO_INCREMENT PRIMARY KEY,
        subject VARCHAR(255),
        predicate VARCHAR(255),
        object TEXT
    )