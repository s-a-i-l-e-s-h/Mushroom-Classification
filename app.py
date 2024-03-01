from flask import Flask, request, render_template
import pickle
app = Flask(__name__)

# change to "redis" and restart to cache again

# some time later
file=open('my_model.pkl','rb')
model=pickle.load(file)
file.close()
      


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #int_features = [int(x) for x in request.form.values()]
    if request.method == 'POST':
        
        cap_shape = request.form["cap-shape"]
        cap_surface = request.form["cap-surface"]
        cap_color = request.form["cap-color"]
        odor = request.form["odor"]
        gill_spacing = request.form["gill-spacing"]
        gill_size = request.form["gill-size"]
        stalk_shape = request.form["stalk-shape"]
        stalk_root = request.form["stalk-root"]
        stalk_surface_above_ring = request.form["stalk-surface-above-ring"]
        stalk_surface_below_ring = request.form["stalk-surface-below-ring"]
        stalk_color_above_ring = request.form["stalk-color-above-ring"]
        stalk_color_below_ring = request.form["stalk-color-below-ring"]
        veil_color = request.form["veil-color"]
        ring_number = request.form["ring-number"]
        spore_print_color = request.form["spore-print-color"]
        population = request.form["population"]

        
        
        data= [[cap_shape,cap_surface,cap_color,odor,gill_spacing,gill_size,stalk_shape,stalk_root,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,veil_color,ring_number,spore_print_color,population]]
        #data = np.array(data)
        #data = data.astype(np.float).reshape(1,-1)
        predict = model.predict(data)
        print(predict)
            
        return render_template('result.html', prediction_text=predict)

    return render_template('index.html')
   


@app.route('/dashboard')
def dashboard():
    
    return render_template('dashboard.html')


if __name__ == "__main__":
    app.run(debug=False,port=5000)
