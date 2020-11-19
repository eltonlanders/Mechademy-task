## The workflow is as follows:
1. First I have cleaned and pre-processed the data. Then I evaluated various ML models and a DL model. <br>
2. Then I have built 2 Flask apps for the models. One to predict the compressor decay state coef and the second to predict the turbine decay state coef. <br>
3. The app has a UI through which one can either input the values separately to make the prediction or a test file can be passed which has to be a sample of the dataframe with __all__ the columns(__excluding__ the compressor decay state coef and turbine decay state coef cols). <br>
4. The app for __compressor decay state coef__ is in __flask_api.py__ and the app for __turbine decay state coef__ is in __flask_api2.py__.
5. To run the apps: <br>
    - Navigate to the working directory in the cmd <br>
    - Enter - __python flask_api.py__ or __python flask_api2.py__ .ex 
    - Navigate to the ip address in the browser
    - After entering the ip address in the browser add __apidocs/__ after it. Example, __http://127.0.0.1:5000/apidocs/__
    - Get predictions.
5. Then I have containerized the applications using Docker. <br>
6. The docker images can be seen in the following image <br>
![images](https://user-images.githubusercontent.com/57378191/99681825-9c498880-2aa4-11eb-904f-c5394a8dfb19.png)
7. The running containers on my Docker desktop as seen in the following image. <br>
![running_containers](https://user-images.githubusercontent.com/57378191/99682559-5a6d1200-2aa5-11eb-9b3b-0bed62635399.png)
8. As it can be seen in the following image, these applications can be executed on any system irrespective of the OS. <br>
![code](https://user-images.githubusercontent.com/57378191/99685109-3b23b400-2aa8-11eb-8f1c-7a5eb0da005e.png)
