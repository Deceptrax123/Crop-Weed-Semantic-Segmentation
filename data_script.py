
def read_file():
    ndvi_train="./data/Sequoia/SequoiaMulti_30/trainNDVI.txt"
    nir_train="./data/Sequoia/SequoiaMulti_30/trainNir.txt" #Contains annotation too
    red_train="./data/Sequoia/SequoiaMulti_30/trainRed.txt" 

    ndvi_test="./data/Sequoia/SequoiaMulti_30/testNDVI.txt"
    nir_test="./data/Sequoia/SequoiaMulti_30/testNir.txt" #Contains annotation too
    red_test="./data/Sequoia/SequoiaMulti_30/testRed.txt"

    #read each file name and extract arrays

    file_paths=[
        ndvi_train,nir_train,red_train,ndvi_test,nir_test,red_test
    ]

    #Train and test split

    train=list()
    test=list()

    #Train file loop
    for i in range(215):
        training_example=list()
        for j in range(3):
            f=open(file_paths[j],'r')
            lines=f.read().splitlines()

            example=lines[i]
            if j==1:
                nir,ground=example.split()[0],example.split()[1]

                training_example.append(nir)
                training_example.append(ground)
            else:
                training_example.append(example)
            
            f.close()
        train.append(training_example)
    
    #test file loop
    for i in range(30):
        testing_example=list()
        for j in range(3):
            k=j+3

            f=open(file_paths[k],'r')
            lines=f.read().splitlines()

            example=lines[i]

            if j==1:
                nir,ground=example.split()[0],example.split()[1]

                testing_example.append(nir)
                testing_example.append(ground)
            else:
                testing_example.append(example)
            f.close()
        test.append(testing_example)

    return train,test