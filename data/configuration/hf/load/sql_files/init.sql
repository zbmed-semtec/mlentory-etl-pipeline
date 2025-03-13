CREATE DATABASE history_DB;
\c history_DB;

CREATE TABLE IF NOT EXISTS "Triplet" (
    "id" BIGSERIAL PRIMARY KEY,
    "subject" VARCHAR(1024) NOT NULL,
    "predicate" VARCHAR(2048) NOT NULL,
    "object" TEXT NOT NULL,
    "triplet_hash" VARCHAR(128) NOT NULL
);

CREATE TABLE IF NOT EXISTS "Triplet_Extraction_Info" (
    "id" BIGSERIAL PRIMARY KEY,
    "method_description" TEXT NOT NULL,
    "extraction_confidence" DECIMAL(6,5)
);

CREATE TABLE IF NOT EXISTS "Version_Range" (
    "id" BIGSERIAL PRIMARY KEY,
    "triplet_id" BIGINT NOT NULL REFERENCES "Triplet"("id") ON DELETE CASCADE,
    "use_start" TIMESTAMP NOT NULL,
    "use_end" TIMESTAMP NOT NULL,
    "deprecated" BOOLEAN NOT NULL DEFAULT FALSE,
    "extraction_info_id" BIGINT NOT NULL REFERENCES "Triplet_Extraction_Info"("id") ON DELETE CASCADE
);

-- Add indexes for frequently queried columns
CREATE INDEX IF NOT EXISTS idx_triplet_subject ON "Triplet" USING hash(subject);
CREATE INDEX IF NOT EXISTS idx_triplet_object_hash ON "Triplet" USING hash(md5(object));
CREATE INDEX IF NOT EXISTS idx_triplet_hash ON "Triplet" USING hash(triplet_hash);

-- Composite index for triplet lookups
CREATE INDEX IF NOT EXISTS idx_triplet_composite ON "Triplet" 
USING btree(subject, predicate, md5(object));

-- Index for extraction info lookups
CREATE INDEX IF NOT EXISTS idx_extraction_info_composite ON "Triplet_Extraction_Info" 
USING btree(method_description, extraction_confidence);

-- Indexes for version range queries
CREATE INDEX IF NOT EXISTS idx_version_range_dates ON "Version_Range" 
USING brin(use_start, use_end);

CREATE INDEX IF NOT EXISTS idx_version_range_triplet ON "Version_Range" 
USING btree(triplet_id);

CREATE INDEX IF NOT EXISTS idx_version_range_deprecated ON "Version_Range" 
USING btree(deprecated);

ALTER TABLE "Version_Range" 
    ADD CONSTRAINT "version_range_triplet_id_foreign" 
    FOREIGN KEY("triplet_id") REFERENCES "Triplet"("id");

ALTER TABLE "Version_Range"
    ADD CONSTRAINT "version_range_extraction_info_id_foreign"
    FOREIGN KEY("extraction_info_id") REFERENCES "Triplet_Extraction_Info"("id");   