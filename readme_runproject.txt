If you want to run this code on your own machine 
1)just adjust all the paths in app.py,main.py in the src file
2)cd to the project directiory
3)update the venv using these in the terminal
		python -m venv venv  # Create a new virtual environment
		.\venv\Scripts\activate  # Activate it
		pip install -r requirements.txt  # Reinstall dependencies
(if you got an error for requirements not found you can still ignore and try to run the code there is a chance it will work)

4)then in the terminal write flask run / go to the src/app and run it mannually
5)go to http://127.0.0.1:5000/
6)load any image from your machine to the website and click on upload
7)after uploading you will see new image with detected objects you can press on anyone to remove from the image using the propper model for detection (cars/general)
8)good luck :)