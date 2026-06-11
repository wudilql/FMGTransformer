# FMGTransformer
The code modification is based on our submitted paper (A Fusion-Guided FMGTransformer Network for Predicting
Methanol Yield in Catalytic CO2 Hydrogenation), which aims to use deep learning algorithms to capture the nonlinear relationships between various reaction characteristics in order to predict methanol yield.

The data used in this project can be obtained from the paper by Manu Suvarna et al.(M. Suvarna, T. P. Araújo, J. Pérez-Ramírez, A generalized machine learning framework to predict the space-time yield of methanol from
thermocatalytic CO2 hydrogenation, Applied Catalysis B: Environmental 315 (2022) 121530. doi:10.1016/j.apcatb.2022.121530.)

# Dependencies of the model
The code is written in python. The implementation of the model relies on the following libraries and tools:
- python 3.9  
- pandas == 2.1.4  
- numpy == 1.26.3  
- pytorch == 2.1.4  
- scikit-learn == 1.3.2  
- keras == 2.7.0  
- matplotlib == 3.8.3

# Future Work
Some simple ideas:
1.The input features can be further improved. In addition to the original variables such as temperature, pressure, feed ratio, and space velocity, more physically and chemically meaningful features can be constructed, such as reciprocal temperature, logarithmic pressure, estimated H2 partial pressure, temperature–pressure interaction terms, and residence-time-related proxy features. These features may help the model better capture the influence of reaction conditions on methanol yield.

2.Chemical prior knowledge and reaction-mechanism constraints can be further introduced into the model. For example, a prior adjacency matrix can be constructed based on the relationships among features such as catalyst type, metal loading, promoter, support, surface area, temperature, pressure, and space velocity, and then used as an attention bias in the model. Meanwhile, constraint terms based on the main reaction of CO2 hydrogenation to methanol and possible side reactions can also be added to the attention mechanism or loss function, so that the prediction is guided not only by data fitting, but also by basic chemical principles.
In short, the hope is to incorporate the chemical mechanism into the model.
<img width="1277" height="1028" alt="image" src="https://github.com/user-attachments/assets/2ee7b19f-94f6-4456-adb5-a37a62cbe4bc" />

