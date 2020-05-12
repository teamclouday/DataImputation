# this file contains codes that test various research targets

from utils.test import *

if __name__ == "__main__":
    dataset_prepare()
    # machine = TestMachine(create_drug_dataset, gen_complete_random, random_gen=1, predictor_cv=5, record_time=True, n_jobs=1, grid_search=False)
    # machine.run()
    # machine.plot_compare_models(save_file_name="drug1.png")
    # machine.plot_compare_completers(save_file_name="drug2.png")
    # drugs = [
    #     "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke",
    #     "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth",
    #     "Mushrooms", "Nicotine", "Semer", "VSA"
    # ]
    # for label in drugs:
    #     machine = BiasDatasetTest(dataset_func=[partial(create_drug_dataset, target_drug=label)], record_time=True, search_best_model=False, predictor_cv=5)
    #     machine.plot_confusion_mat(savefig=True)
    #machine = BiasDatasetTest(record_time=True, search_best_model=False)
    #machine.plot_confusion_mat_protected(savefig=True)