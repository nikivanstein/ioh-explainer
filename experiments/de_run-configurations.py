from config import de_explainer

de_explainer.run(paralell=True, start_index=0, checkpoint_file="intermediate_de_30d.csv")
de_explainer.save_results("de_final_30d.pkl")
