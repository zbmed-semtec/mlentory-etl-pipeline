CREATE DATABASE history_DB;
\c history_DB;

CREATE TABLE IF NOT EXISTS "Triplet" (
    "id" BIGSERIAL PRIMARY KEY,
    "subject" VARCHAR(1024) NOT NULL,
    "predicate" VARCHAR(2048) NOT NULL,
    "object" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "Triplet_Extraction_Info" (
    "id" BIGSERIAL PRIMARY KEY,
    "method_description" TEXT NOT NULL,
    "extraction_confidence" DECIMAL(6,5),
    "platform" VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS "Version_Range" (
    "id" BIGSERIAL PRIMARY KEY,
    "triplet_id" BIGINT NOT NULL REFERENCES "Triplet"("id") ON DELETE CASCADE,
    "use_start" TIMESTAMP NOT NULL,
    "use_end" TIMESTAMP NOT NULL,
    "deprecated" BOOLEAN NOT NULL DEFAULT FALSE,
    "extraction_info_id" BIGINT NOT NULL REFERENCES "Triplet_Extraction_Info"("id") ON DELETE CASCADE
);

ALTER TABLE "Version_Range" 
    ADD CONSTRAINT "version_range_triplet_id_foreign" 
    FOREIGN KEY("triplet_id") REFERENCES "Triplet"("id");

ALTER TABLE "Version_Range"
    ADD CONSTRAINT "version_range_extraction_info_id_foreign"
    FOREIGN KEY("extraction_info_id") REFERENCES "Triplet_Extraction_Info"("id");   