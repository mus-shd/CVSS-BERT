import numpy as np
import unittest

test_data = ['A vulnerability in Cisco IOS ROM Monitor (ROMMON) Software for Cisco Catalyst 6800 Series Switches could allow an unauthenticated, local attacker to bypass Cisco Secure Boot validation checks and load a compromised software image on an affected device. The vulnerability is due to the presence of a hidden command in the affected software. An attacker could exploit this vulnerability by connecting to an affected device via the console, forcing the device into ROMMON mode, and writing a malicious pattern to a specific memory address on the device. A successful exploit could allow the attacker to bypass signature validation checks by Cisco Secure Boot technology and load a compromised software image on the affected device. A compromised software image is any software image that has not been digitally signed by Cisco.',
 'The official haproxy docker images before 1.8.18-alpine (Alpine specific) contain a blank password for a root user. System using the haproxy docker container deployed by affected versions of the docker image may allow a remote attacker to achieve root access with a blank password.',
 'Lack of administrator control over security vulnerability in client.cgi in Synology SSL VPN Client before 1.2.5-0226 allows remote attackers to conduct man-in-the-middle attacks via the (1) command, (2) hostname, or (3) port parameter.',
 'An issue was discovered in S-CMS v1.5. There is an XSS vulnerability in search.php via the keyword parameter.',
 'In Audio File Library (aka audiofile) 0.3.6, there exists one NULL pointer dereference bug in ulaw2linear_buf in G711.cpp in libmodules.a that allows an attacker to cause a denial of service via a crafted file.',
 'On BIG-IP 14.1.0-14.1.2.3, 14.0.0-14.0.1, 13.1.0-13.1.3.1, and 12.1.0-12.1.4.1, when processing TLS traffic with hardware cryptographic acceleration enabled on platforms with Intel QAT hardware, the Traffic Management Microkernel (TMM) may stop responding and cause a failover event.',
 'A user authorized to perform database queries may trigger denial of service by issuing specially crafted queries, which use the $mod operator to overflow negative values. This issue affects: MongoDB Inc. MongoDB Server v4.4 versions prior to 4.4.1; v4.2 versions prior to 4.2.9; v4.0 versions prior to 4.0.20; v3.6 versions prior to 3.6.20.',
 'Adobe Acrobat and Reader versions 2020.006.20042 and earlier, 2017.011.30166 and earlier, 2017.011.30166 and earlier, and 2015.006.30518 and earlier have an out-of-bounds write vulnerability. Successful exploitation could lead to arbitrary code execution .',
 'In rw_i93_sm_detect_ndef of rw_i93.c, there is a possible information disclosure due to a missing bounds check. This could lead to remote information disclosure with no additional execution privileges needed. User interaction is not needed for exploitation.',
 'Parsing malformed project files in Omron CX-One versions 4.42 and prior, including the following applications: CX-FLnet versions 1.00 and prior, CX-Protocol versions 1.992 and prior, CX-Programmer versions 9.65 and prior, CX-Server versions 5.0.22 and prior, Network Configurator versions 3.63 and prior, and Switch Box Utility versions 1.68 and prior, may allow the pointer to call an incorrect object resulting in an access of resource using incompatible type condition.']





from data import train_test_LabelEncoder, tokenizer


class TestData(unittest.TestCase):
    def test_LabelEncoder(self):
        """
        Test LabelEncoder works properly
        """
        train_labels = ['HIGH', 'HIGH', 'HIGH', 'LOW', 'NONE', 'NONE', 'NONE', 'HIGH', 'HIGH', 'HIGH']
        test_labels = ['LOW', 'HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH', 'HIGH', 'HIGH', 'HIGH', 'HIGH']
        encoded_train_labels, encoded_test_labels = train_test_LabelEncoder(train_labels, test_labels)
        
        self.assertEqual(len(train_labels), len(encoded_train_labels))
        self.assertEqual(len(test_labels), len(encoded_test_labels))
        
        expected_encoded_train = [0, 0, 0, 1, 2, 2, 2, 0, 0, 0]
        expected_encoded_test = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
        self.assertListEqual(encoded_train_labels.tolist(), expected_encoded_train)
        self.assertListEqual(encoded_test_labels.tolist(), expected_encoded_test)
        
        
    def test_tokenizer(self):
        """
        Test predict works properly
        """
        myTokenizer = tokenizer()
        
        tokenized_batch = myTokenizer(test_data, truncation=True, padding=True, max_length=128)
        single_sample = myTokenizer(test_data[0], truncation=True, padding=True, max_length=128)
        
        tokenized_batch_shape = np.array(tokenized_batch['input_ids']).shape
        single_sample_shape = np.array(single_sample['input_ids']).shape
        #print("batch shape:", tokenized_batch_shape)
        #print("sample shape:", single_sample_shape)
        
        #ensure the tokenized set contain the same number of element
        self.assertEqual(len(test_data), tokenized_batch_shape[0])
        
        

from model import BertClassifier

classifier =  BertClassifier()



class TestModel(unittest.TestCase):
    def test_predict(self):
        """
        Test predict works properly
        """
        myTokenizer = tokenizer()
        
        tokenized_batch = myTokenizer(test_data, truncation=True, padding=True, max_length=128)
        single_sample = myTokenizer(test_data[0], truncation=True, padding=True, max_length=128)
        
        
        tokenized_batch_shape = np.array(tokenized_batch['input_ids']).shape
        single_sample_shape = np.array(single_sample['input_ids']).shape
        #print("batch shape:", tokenized_batch_shape)
        #print("sample shape:", single_sample_shape)
        
        batch_predictions = classifier.predict(tokenized_batch)
        single_prediction = classifier.predict(single_sample)
        
        #ensure the number of predictions is equal to the number of instances in the batch
        self.assertEqual(tokenized_batch_shape[0], batch_predictions['predicted_labels'].size(0))
        
        #ensure that the function return the same result whether an instance is fed alone or through a batch
        self.assertEqual(batch_predictions['predicted_labels'][0].item(), single_prediction['predicted_labels'].item())
        self.assertAlmostEqual(batch_predictions['predicted_scores'][0].item(), single_prediction['predicted_scores'].item(), 4)
        
        

        