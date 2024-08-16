CREATE DATABASE IF NOT EXISTS MLentory_DB;
CREATE DATABASE IF NOT EXISTS test_DB;
USE MLentory_DB;

CREATE TABLE IF NOT EXISTS `Version_Range`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `triplet_id` BIGINT NOT NULL,
    `start` DATETIME NOT NULL,
    `end` DATETIME NOT NULL,
    `extraction_info_id` BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS `Triplet`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `subject` TEXT NOT NULL,
    `relation` TEXT NOT NULL,
    `object` TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS `Triplet_Extraction_Info`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `method_id` BIGINT NOT NULL,
    `extraction_confidence` BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS `Method`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` TEXT NOT NULL,
    `description` TEXT NOT NULL
);
ALTER TABLE
    `Triplet_Extraction_Info` ADD CONSTRAINT `triplet_extraction_info_method_id_foreign` FOREIGN KEY(`method_id`) REFERENCES `Method`(`id`);
ALTER TABLE
    `Version_Range` ADD CONSTRAINT `version_range_triplet_id_foreign` FOREIGN KEY(`triplet_id`) REFERENCES `Triplet`(`id`);
ALTER TABLE
    `Version_Range` ADD CONSTRAINT `version_range_extraction_info_id_foreign` FOREIGN KEY(`extraction_info_id`) REFERENCES `Triplet_Extraction_Info`(`id`);