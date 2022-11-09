from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import csv

class database_operations:
    """
        Description : This class will be used for connecting and operating the database functions and stuff. Like, connection
        to the databse, inserting, extracting data etc.

        Database : Cassandra
        Vresion : 1.0

    """

    def __init__(self, logger, path):
        try:
            self.log = logger
            self.path = path

            cloud_config = {
                 'secure_connect_bundle': './secure-connect-ineuron.zip'
             }
            auth_provider = PlainTextAuthProvider('jmQhBvNrCaxwBrjyIWRFCHrk',
                                                   'nY575HZTOQ4TEmRRqR5EUqqeqZUZgQ5kkxKDiFlzdjjI_0PuzIWumorIPJWeK2GNf8a1,iYXFZA.-FIT7Q67IvH+acparkG2BqfjpZtYnh5ION6ANxedCLTIsrujhIng')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            self.session = cluster.connect()

            self.session.execute(""" CREATE TABLE if not exists ineuron.spam_training_dataset(
                     sample_text text PRIMARY KEY,output_str text,output_num int
                     );""")

            self.session.execute('use ineuron;')
            self.log.info('Database Updated Successfully.')

        except Exception as e:
            self.log.warning('An error has occurred : {}'.format(e))
            raise Exception

    def uploading_data(self):
        try:
            csv_reader = csv.reader(open(self.path))
            next(csv_reader)
            # Uploading the dataset in the database.
            
            for rows in csv_reader:
                query = """INSERT INTO ineuron.spam_dataset(sample_text,output_str,output_num)
                VALUES('%s','%s','%d');""" % (
                     str(rows[1]), str(rows[2]), int(rows[3]))
                self.session.execute(query)
            self.log.info('Dataset Uploaded Successfully.')

        except Exception as e:
            self.log.warning('An error has occurred : {}'.format(e))
            raise Exception

    def extracting_data(self):
        try:
            # Getting the training file ready.
            dataset_path = '/config/workspace/Spam_Ham_Classifier/dataset_spam.csv'
            training = open(dataset_path, 'w', encoding='utf8')
            training.truncate()
            write = csv.writer(training)
            
            # Writing the column names of the dataset.
            write.writerow(tuple(['sample_text', 'output_str', 'output_num']))
            for val in self.session.execute('select * from ineuron.spam_dataset;'):
                 row = [val.sample_text, val.output_str, val.output_num]
                 # Putting all the rows in the csv file
                 write.writerow(tuple(row))
            
            training.close()
            self.log.info('Operation Successful. {}'.format(main_train_data_path))
            return main_train_data_path

        except Exception as e:
            self.log.warning('An errorr has occurred : {}'.format(e))
            raise Exception

    def start_db(self):
        self.uploading_data()
        train_path = self.extracting_data()
        return train_path