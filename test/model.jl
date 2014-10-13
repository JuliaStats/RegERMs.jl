X = [1 1; 2 2; 1 -1]
y = [1; 1; 2]

model = Model(X, y, :multinomial)
@test predict(model, X) == y
@test values(model, X) == [3.0 6.0 -2.0]'
