from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def slimshady():
    return "Uh, sama lamaa duma lamaa you assuming I'm a human What I gotta do to get it through to you I'm superhuman Innovative and I'm made of rubber  So that anything you say is ricocheting off of me and it'll glue to you  I'm never stating, more than ever demonstrating  How to give a motherf---in' audience a feeling like it's levitating  Never fading, and I know the haters are forever waiting  For the day that they can say I fell off, they'd be celebrating  Cause I know the way to get 'em motivated  I make elevating music!"


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('uploads/uploaded_file.jpg')


