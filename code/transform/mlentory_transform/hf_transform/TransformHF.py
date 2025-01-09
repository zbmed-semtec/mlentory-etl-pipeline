import pandas as pd
from tqdm import tqdm
from .FieldProcessorHF import FieldProcessorHF

class TransformHF:
    def __init__(self, new_schema: pd.DataFrame, transformations: pd.DataFrame):
        """
        Initialize the HuggingFace transformer with schema and transformations.
        
        Args:
            new_schema (pd.DataFrame): DataFrame containing the target schema
            transformations (pd.DataFrame): DataFrame containing column transformations
        """
        self.new_schema = new_schema
        self.transformations = transformations
        self.fields_processor = FieldProcessorHF(self.new_schema, self.transformations)

    def transform(
        self, 
        extracted_df: pd.DataFrame, 
        save_output: bool = False,
        output_dir: str = None
    ) -> pd.DataFrame:
        """
        Transform the extracted data into the target schema.
        
        Args:
            extracted_df (pd.DataFrame): DataFrame containing extracted model data
            save_output (bool, optional): Whether to save the transformed data. Defaults to False.
            output_dir (str, optional): Directory to save the transformed data. Required if save_output is True.
            
        Returns:
            pd.DataFrame: Transformed DataFrame conforming to the target schema
            
        Raises:
            ValueError: If save_output is True but output_dir is not provided
        """
        if (save_output==True) and (output_dir is None):
            raise ValueError("output_dir must be provided when save_output is True")
        
        processed_models = []
        
        for row_num, row in tqdm(
            extracted_df.iterrows(), 
            total=len(extracted_df), 
            desc="Transforming progress"
        ):
            model_data = self.fields_processor.process_row(row)
            processed_models.append(model_data)
        
        transformed_df = pd.DataFrame(list(processed_models))
        
        if save_output:
            output_path = f"{output_dir}/transformation_results.csv"
            transformed_df.to_csv(output_path, index=False)
            print(f"Saved transformation results to {output_path}")
        
        return transformed_df
