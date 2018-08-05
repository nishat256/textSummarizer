from flask import Flask,render_template,request
from main import return_summary
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('home.html')
@app.route('/result',methods=['POST'])
def processText():
	bigText=request.form['bigText'].strip()
	SUMMARY_SIZE=int(request.form['num'])
	if len(bigText)==0:
		return 'Please paste some text in the box'
	resp=return_summary(bigText,SUMMARY_SIZE)
	if not resp[0]:
		return resp[1]
	else:
		return render_template('result.html',data=(bigText,resp[1]))
if __name__ == '__main__':
   app.run(debug=True,threaded=True)