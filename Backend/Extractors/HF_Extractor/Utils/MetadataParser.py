class MetadataParser:

    def __init__(self,qa_pipeline):
        self.qa_pipeline = qa_pipeline
    
    def answer_question(self,question, context):
        answer = self.qa_pipeline({"question": question, "context": context})
        print("Question:", question)
        print("Answer:", answer,"/n")
        return answer['answer']+''
    
    def parse_known_fields_HF(self,HF_df):
        HF_df.loc[:,"q_id_0"] = HF_df.loc[:, ("modelId")]
        HF_df.loc[:,"q_id_1"] = HF_df.loc[:, ("author")]
        HF_df.loc[:,"q_id_2"] = HF_df.loc[:, ("createdAt")]
        return HF_df