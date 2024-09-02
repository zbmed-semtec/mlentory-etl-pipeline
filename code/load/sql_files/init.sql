CREATE DATABASE IF NOT EXISTS MLentory_DB;
USE MLentory_DB;

CREATE TABLE IF NOT EXISTS `Version_Range`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `triplet_id` BIGINT NOT NULL,
    `start` DATETIME NOT NULL,
    `end` DATETIME NOT NULL,
    `deprecated` BOOLEAN NOT NULL DEFAULT FALSE,
    `extraction_info_id` BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS `Triplet`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `subject` TEXT NOT NULL,
    `relation` TEXT NOT NULL,
    `object` TEXT NOT NULL,
    UNIQUE KEY `unique_triplet` (`subject`(512), `relation`(512), `object`(512))
);
CREATE TABLE IF NOT EXISTS `Triplet_Extraction_Info`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `method_description` BIGINT NOT NULL,
    `extraction_confidence` FLOAT NOT NULL
);

ALTER TABLE
    `Triplet_Extraction_Info` ADD CONSTRAINT `triplet_extraction_info_method_id_foreign` FOREIGN KEY(`method_id`) REFERENCES `Method`(`id`);
ALTER TABLE
    `Version_Range` ADD CONSTRAINT `version_range_triplet_id_foreign` FOREIGN KEY(`triplet_id`) REFERENCES `Triplet`(`id`);
ALTER TABLE
    `Version_Range` ADD CONSTRAINT `version_range_extraction_info_id_foreign` FOREIGN KEY(`extraction_info_id`) REFERENCES `Triplet_Extraction_Info`(`id`);