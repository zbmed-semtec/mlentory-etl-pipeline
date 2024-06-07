# Transform

Here you can find the code and design decisions for the transform stage.

## Overview
The files that the transform stage will be processing are stored in the [Transform_Queue](https://github.com/zbmed-semtec/mlentory-etl-pipeline/tree/main/code/Transform_Queue) folder.
The transform stage keeps track of the Transform_queue folder using the [QueueObserver](https://github.com/zbmed-semtec/mlentory-etl-pipeline/blob/main/code/Transform/Core/QueueObserver.py) class. If a new file is created or deleted on the folder the QueueObserver gets notified of the event.

When a new file is created the QueueObserver sends the path of the new file to the [FileProcessor](https://github.com/zbmed-semtec/mlentory-etl-pipeline/blob/main/code/Transform/Core/FilesProcessor.py) class. The FileProcessor can process files in parallel, but it has a limited amount of parallel processes it can spawn. There are two ways in which it will start processing files:
1. The amount of files is equal to the amount of parallel processes available.
2. An amount of files less than the amount of parallel processes available but greater than 0 has been waiting to be processed longer than the established waiting period.

## Logging
The log files , that record everything that happens during the execution of the project, can be found in the folder [Processing_Logs]()
A new log file is created every time a new instance of the Transform stage is created. 