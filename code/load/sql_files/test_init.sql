
CREATE DATABASE test_DB;
\c test_DB;

-- CREATE USER 'test_user'@'%' IDENTIFIED BY 'test_pass';
-- GRANT ALL PRIVILEGES ON test_DB.* TO 'test_user'@'%';

-- CREATE USER 'additional_user1'@'%' IDENTIFIED BY 'password1';
-- GRANT ALL PRIVILEGES ON test_DB.* TO 'additional_user1'@'%';

-- CREATE USER 'additional_user2'@'%' IDENTIFIED BY 'password2';
-- GRANT ALL PRIVILEGES ON test_DB.* TO 'additional_user2'@'%';

-- FLUSH PRIVILEGES;

CREATE TABLE IF NOT EXISTS "Version_Range" (
    "id" BIGSERIAL PRIMARY KEY,
    "triplet_id" BIGINT NOT NULL,
    "use_start" TIMESTAMP NOT NULL,
    "use_end" TIMESTAMP NOT NULL,
    "deprecated" BOOLEAN NOT NULL DEFAULT FALSE,
    "extraction_info_id" BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS "Triplet" (
    "id" BIGSERIAL PRIMARY KEY,
    "subject" VARCHAR(1024) NOT NULL,
    "predicate" VARCHAR(1024) NOT NULL,
    "object" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "Triplet_Extraction_Info" (
    "id" BIGSERIAL PRIMARY KEY,
    "method_description" TEXT NOT NULL,
    "extraction_confidence" DECIMAL(6,5)
);

ALTER TABLE "Version_Range" 
    ADD CONSTRAINT "version_range_triplet_id_foreign" 
    FOREIGN KEY("triplet_id") REFERENCES "Triplet"("id");

ALTER TABLE "Version_Range"
    ADD CONSTRAINT "version_range_extraction_info_id_foreign"
    FOREIGN KEY("extraction_info_id") REFERENCES "Triplet_Extraction_Info"("id");   

