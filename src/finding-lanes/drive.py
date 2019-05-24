from flask import Flask
# Flask: microframework to build web apps
# * __name__ : WWhenever you execture python script, python assigns the name main
# 

app = Flask(__name__ ) #'__main__'

# * Tells Flask what url we should use to trigger
# greeting function
@app.route('/hpome')

def greeting():
	return "Welcome!"


if __name__ == '__main__':
	# if we execute the scipt we want to run the application
	app.run(port = 3000)