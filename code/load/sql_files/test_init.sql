
CREATE DATABASE IF NOT EXISTS test_DB;
USE test_DB;

-- CREATE USER 'test_user'@'%' IDENTIFIED BY 'test_pass';
-- GRANT ALL PRIVILEGES ON test_DB.* TO 'test_user'@'%';

-- CREATE USER 'additional_user1'@'%' IDENTIFIED BY 'password1';
-- GRANT ALL PRIVILEGES ON test_DB.* TO 'additional_user1'@'%';

-- CREATE USER 'additional_user2'@'%' IDENTIFIED BY 'password2';
-- GRANT ALL PRIVILEGES ON test_DB.* TO 'additional_user2'@'%';

-- FLUSH PRIVILEGES;

CREATE TABLE IF NOT EXISTS `Version_Range`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `triplet_id` BIGINT UNSIGNED NOT NULL,
    `start` DATETIME NOT NULL,
    `end` DATETIME NOT NULL,
    `deprecated` BOOLEAN NOT NULL DEFAULT FALSE,
    `extraction_info_id` BIGINT UNSIGNED NOT NULL
);
CREATE TABLE IF NOT EXISTS `Triplet`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `subject` varchar(512) NOT NULL,
    `predicate` varchar(512) NOT NULL,
    `object` TEXT NOT NULL,
    UNIQUE KEY `unique_triplet` (`subject`(512), `predicate`(512), `object`(512))
);
CREATE TABLE IF NOT EXISTS `Triplet_Extraction_Info`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `method_description` TEXT NOT NULL,
    `extraction_confidence` FLOAT NOT NULL
);

-- For the first constraint
-- IF NOT EXISTS (
--     SELECT 1 FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
--     WHERE CONSTRAINT_NAME = 'version_range_triplet_id_foreign'
-- ALTER TABLE `Version_Range` DROP CONSTRAINT IF EXISTS ( SELECT 1 FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
--                                                         WHERE CONSTRAINT_NAME = 'version_range_triplet_id_foreign');

-- ALTER TABLE `Version_Range`
-- ADD CONSTRAINT `version_range_triplet_id_foreign` 
-- FOREIGN KEY(`triplet_id`) REFERENCES `Triplet`(`id`);


-- For the second constraint
-- IF NOT EXISTS (
--     SELECT 1 FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
--     WHERE CONSTRAINT_NAME = 'version_range_extraction_info_id_foreign'
-- ) THEN
-- ALTER TABLE `Version_Range` DROP CONSTRAINT IF EXISTS `version_range_extraction_info_id_foreign`;

-- ALTER TABLE `Version_Range` 
-- ADD CONSTRAINT `version_range_extraction_info_id_foreign` 
-- FOREIGN KEY(`extraction_info_id`) REFERENCES `Triplet_Extraction_Info`(`id`);

