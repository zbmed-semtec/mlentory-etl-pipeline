CREATE TABLE `models`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL,
    `extraction_date` JSON NOT NULL,
    `hf_user_id` BIGINT NOT NULL,
    `base_model` JSON NOT NULL,
    `Hyperparameters` JSON NOT NULL,
    `Hyperparameters_values` JSON NOT NULL,
    `deployed_at` JSON NOT NULL,
    `authors` JSON NOT NULL
);
CREATE TABLE `hf_users`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL
);
CREATE TABLE `models_tasks`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `task_id` BIGINT NOT NULL,
    `model_id` BIGINT NOT NULL
);
CREATE TABLE `models_licences`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `licence_id` JSON NOT NULL,
    `model_id` JSON NOT NULL
);
CREATE TABLE `tasks`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL
);
CREATE TABLE `models_publications`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `model_id` BIGINT NOT NULL,
    `publication_id` BIGINT NOT NULL
);
CREATE TABLE `models_datasets`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `dataset_id` BIGINT NOT NULL,
    `model_id` BIGINT NOT NULL,
    `propouse` MEDIUMTEXT NOT NULL,
    `evaluation_metrics` JSON NOT NULL,
    `evaluation_metrics_results` JSON NOT NULL
);
CREATE TABLE `datasets`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL
);
CREATE TABLE `libraries`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL
);
CREATE TABLE `models_libraries`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `library_id` BIGINT NOT NULL,
    `model_id` BIGINT NOT NULL
);
CREATE TABLE `Publications`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL,
    `publisher` MEDIUMTEXT NOT NULL
);
CREATE TABLE `licences`(
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `name` JSON NOT NULL,
    `description` MEDIUMTEXT NOT NULL
);
ALTER TABLE
    `models_tasks` ADD CONSTRAINT `models_tasks_model_id_foreign` FOREIGN KEY(`model_id`) REFERENCES `models`(`id`);
ALTER TABLE
    `models_publications` ADD CONSTRAINT `models_publications_model_id_foreign` FOREIGN KEY(`model_id`) REFERENCES `models`(`id`);
ALTER TABLE
    `models_publications` ADD CONSTRAINT `models_publications_publication_id_foreign` FOREIGN KEY(`publication_id`) REFERENCES `Publications`(`id`);
ALTER TABLE
    `models_licences` ADD CONSTRAINT `models_licences_licence_id_foreign` FOREIGN KEY(`licence_id`) REFERENCES `licences`(`id`);
ALTER TABLE
    `models_datasets` ADD CONSTRAINT `models_datasets_model_id_foreign` FOREIGN KEY(`model_id`) REFERENCES `models`(`id`);
ALTER TABLE
    `models_libraries` ADD CONSTRAINT `models_libraries_library_id_foreign` FOREIGN KEY(`library_id`) REFERENCES `models`(`id`);
ALTER TABLE
    `models` ADD CONSTRAINT `models_hf_user_id_foreign` FOREIGN KEY(`hf_user_id`) REFERENCES `hf_users`(`id`);
ALTER TABLE
    `models_tasks` ADD CONSTRAINT `models_tasks_task_id_foreign` FOREIGN KEY(`task_id`) REFERENCES `tasks`(`id`);
ALTER TABLE
    `models_libraries` ADD CONSTRAINT `models_libraries_library_id_foreign` FOREIGN KEY(`library_id`) REFERENCES `libraries`(`id`);
ALTER TABLE
    `models_licences` ADD CONSTRAINT `models_licences_model_id_foreign` FOREIGN KEY(`model_id`) REFERENCES `models`(`id`);
ALTER TABLE
    `models_datasets` ADD CONSTRAINT `models_datasets_dataset_id_foreign` FOREIGN KEY(`dataset_id`) REFERENCES `datasets`(`id`);