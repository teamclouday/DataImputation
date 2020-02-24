# this file contains codes that test various research targets

from utils.test import *

if __name__ == "__main__":
    dataset_prepare()
    # machine = TestMachine(create_iris_dataset, gen_complete_random, random_gen=10, predictor_cv=10, record_time=True)
    # machine.run()
    # machine.plot_compare_models(save_file_name="iris1.png")
    # machine.plot_compare_completers(save_file_name="iris2.png")
    # machine = TestMachine(create_bank_dataset, gen_complete_random, random_gen=5, predictor_cv=10, record_time=True)
    # machine.run()
    # machine.plot_compare_models(save_file_name="bank1.png")
    # machine.plot_compare_completers(save_file_name="bank2.png")
    machine = TestMachine(create_heart_dataset, gen_complete_random, random_gen=10, predictor_cv=10, record_time=True)
    machine.run()
    machine.plot_compare_models(save_file_name="heart1.png")
    machine.plot_compare_completers(save_file_name="heart2.png")