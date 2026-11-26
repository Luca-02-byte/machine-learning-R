#*****************************************************************************
#  ML PROJECT - Luca-02-byte
#*****************************************************************************
# Merge applicato dei lab di ML:
# *   regressione (polinomiale, KNN, alberi, bagging, boosting, spline)
# *   classificazione (logistica, LDA, QDA, Naive Bayes, KNN, alberi, random forest)
# *   non supervisionato (clustering gerarchico, K-means, PCA)
# Fonte: https://mcameletti.quarto.pub/mlfe-25-26-rlabs/


#*****************************************************************************
#  LIBRARIES
#*****************************************************************************
library(MASS)
library(splines)
library(tidyverse)
library(tidymodels)
library(kknn)
library(discrim)      
library(pROC)     
library(e1071)        
library(GGally)       
library(rpart.plot)   
library(randomForest) 
library(gbm)          
library(ggfortify)    
library(factoextra)   
library(ggcorrplot)   


#*****************************************************************************
# REGRESSIONE -> BOSTON DATA
#*****************************************************************************

glimpse(Boston)

summary(Boston$medv)
Boston %>%
  ggplot()+
  geom_histogram(aes(medv),
                 bins=10, col="black", fill="deepskyblue")

Boston %>%
  cor() %>%
  ggcorrplot(lab = TRUE,
             lab_size=2)

cor(Boston)[,"medv"] %>%
  sort()

# Istat e rm (numero medio di stanze, correlazione positiva) sono le variabili piu' legate a medv. Le visualizziamo insieme colorando i punti con la risposta
Boston %>%
  ggplot()+
  geom_point(aes(x=lstat,y=rm,col=medv))+
  scale_colour_gradientn(colours = rev(hcl.colors(12)))

# Le zone piu' care stanno in alto a sinistra: poche persone di status basso e case con piu' stanze.  Relazione tra lstat e medv, con la curva loess di geom_smooth 
Boston %>%
  ggplot(aes(x=lstat, y=medv)) +
  geom_point() +
  geom_smooth()

# La relazione non e' lineare: la curva scende ripida per valori bassi di lstat e poi si appiattisce. Ci aspettiamo che i modelli più flessibili battano la retta.


#*****************************************************************************
#*  TRAINING, VALIDATION E TEST SET (METODO 1)
#*****************************************************************************
#  70% training, 15% validation, 15% test, campionando un'etichetta per ogni
#  Prima creiamo lstat_ord, la versione a 4 classi di lstat per le step functions.

# Step functions need a qualitative version of lstat: 4 classes of equal length
Boston$lstat_ord = cut(Boston$lstat, breaks = 4)
table(Boston$lstat_ord)

# Regressors used by the models with all the variables (lstat_ord is excluded)
vars_all = setdiff(colnames(Boston), "lstat_ord")

set.seed(1, sample.kind="Rejection")
# Sample the indexes: 1 = training, 2 = validation, 3 = test
index = sample(c(1,2,3),
               size=nrow(Boston),
               prob=c(0.7,0.15,0.15),
               replace=T) 

boston_training = Boston[index==1,]
boston_validation = Boston[index==2,]
boston_test = Boston[index==3,]

c(training = nrow(boston_training),
  validation = nrow(boston_validation),
  test = nrow(boston_test))


#*****************************************************************************
#*  TUNING SUL VALIDATION SET
#*****************************************************************************
# Due modelli hanno un parametro di complessita': il grado del polinomio d e il numero di vicini K. Li tariamo prima di toccare il test set.

#=============================== Grado del polinomio ==========================

# Linear regression specification 
lm_spec = linear_reg() %>%
  set_engine("lm")

# Create an empty vector where values of the MSE will be saved
mse_poly_vec = c()

# Create a vector with the values of d
d = 1:6 #polynomial order from 1 to 6

# Start the for loop
for (i in 1:length(d)) {
  # Fit the model
  poly_fit = lm_spec %>%
    fit(medv ~ poly(lstat, d[i]), data = boston_training)

  # Make predictions on the validation data
  poly_preds = poly_fit %>%
    predict(new_data = boston_validation)

  # Calculate the validation MSE
  mse_poly_vec[i] = mean((boston_validation$medv - poly_preds$.pred)^2)
}

# The vector of the validation MSEs
mse_poly_vec

data.frame(d, mse_poly_vec) %>%
  ggplot(aes(d, mse_poly_vec)) +
  geom_point()+
  geom_line()

best_poly = d[which.min(mse_poly_vec)] # tuned degree
best_poly

# Il grado che minimizza il validation MSE e' best_poly. Il salto piu'
# grande si ha passando dalla retta (d=1) al grado 2; dopo i miglioramenti sono molto piu' piccoli.

#=============================== Numero di vicini =============================
# Per il KNN usiamo TUTTE le variabili. Il motore kknn standardizza di default i regressori numerici, quindi le unita' di misura non dominano le distanze.

# Create a vector with the values of K
K_vec = c(1, 3, 5, 10, 15, 20, 25, 50, 100)

# Create an empty vector where values of the MSE will be saved
mse_knn_vec = c()

# Start the for loop
for (i in 1:length(K_vec)) {

  # Define the KNN model specification
  knn_spec = nearest_neighbor(neighbors = K_vec[i], weight_func = "rectangular") %>%
    set_engine("kknn") %>%
    set_mode("regression")

  # Fit the KNN model
  knn_fit = knn_spec %>%
    fit(medv ~ ., data = boston_training[, vars_all])

  # Make predictions on the validation data
  knn_preds = knn_fit %>%
    predict(new_data = boston_validation[, vars_all])

  # Calculate the validation MSE
  mse_knn_vec[i] = mean((boston_validation$medv - knn_preds$.pred)^2)
}

mse_knn_vec

# high to low flexibility
data.frame(K_vec, mse_knn_vec) %>%
  ggplot(aes(K_vec, mse_knn_vec)) +
  geom_point()+
  geom_line()

K_reg = K_vec[which.min(mse_knn_vec)] # tuned number of neighbours
K_reg

# Il grafico va letto da sinistra (K piccolo, modello molto flessibile) a destra (K grande, poco flessibile): il validation error ha tipicamente una forma a U.

#=========================== Riuniamo training e validation ===================
# Scelti d e K non serve piu' un validation set separato: lo riuniamo al training per avere piu' dati nella stima finale.

boston_trainval = rbind(boston_validation, boston_training)
dim(boston_trainval)

#*****************************************************************************
#*  UNA SOLA VARIABILE: LSTAT
#*****************************************************************************
# Confrontiamo cinque modi di descrivere la relazione tra
# lstat e medv: retta, funzioni a gradino, polinomio (grado tarato), spline cubica e spline cubica naturale. Nodi = tre quartili di lstat.

# Linear model
mod_lm = lm(medv ~ lstat,
            data = boston_trainval)

# Step functions: one constant for each class of lstat_ord
mod_cut = lm(medv ~ lstat_ord,
             data = boston_trainval)

# Polynomial regression with the tuned degree
mod_poly = lm(medv ~ poly(lstat, best_poly),
              data = boston_trainval)

# Cubic spline and natural cubic spline, knots = quartiles of lstat
knots = quantile(boston_trainval$lstat,
                 probs = c(0.25, 0.5, 0.75))
mod_cs = lm(medv ~ bs(lstat, knots = knots, degree = 3),
            data = boston_trainval)
mod_ns = lm(medv ~ ns(lstat, knots = knots),
            data = boston_trainval)

summary(mod_lm)
coef(mod_cut) # the intercept is the mean of medv in the first class of lstat_ord

widedata = boston_trainval %>%
  mutate(pred_lm = mod_lm$fitted.values,
         pred_step = mod_cut$fitted.values,
         pred_poly = mod_poly$fitted.values,
         pred_cubic = mod_cs$fitted.values,
         pred_natural = mod_ns$fitted.values)

longdata = widedata %>%
  pivot_longer(cols = pred_lm:pred_natural,
               names_to = "model",
               values_to = "pred")

longdata %>%
  ggplot(aes(lstat, medv)) +
  geom_point(col="gray") +
  geom_line(aes(lstat, pred, col=model), lwd=1.2) +
  facet_wrap(~model) +
  theme(legend.position = "none")

# La retta non segue la curvatura, i gradini la approssimano a scalini, polinomio e spline sono molto vicini tra loro.
# Valutiamo tutto sul TEST SET con MSE e correlazione osservati/previsti.

# function computing test MSE and correlation between predicted and observed values
perf = function(pred, obs){
  return(data.frame(MSE = mean((pred - obs)^2),
                    COR = cor(pred, obs)))
}

lm_pred = predict(mod_lm, boston_test)
cut_pred = predict(mod_cut, boston_test)
poly_pred = predict(mod_poly, boston_test)
cs_pred = predict(mod_cs, boston_test)
ns_pred = predict(mod_ns, boston_test)

perf_lstat = rbind(lm_lstat = perf(lm_pred, boston_test$medv),
                   step_lstat = perf(cut_pred, boston_test$medv),
                   poly_lstat = perf(poly_pred, boston_test$medv),
                   cubic_lstat = perf(cs_pred, boston_test$medv),
                   natural_lstat = perf(ns_pred, boston_test$medv))

perf_lstat %>%
  rownames_to_column("model") %>%
  arrange(MSE)

# modello migliore con la sola lstat e confronto con la retta
rownames(perf_lstat)[which.min(perf_lstat$MSE)]
round(min(perf_lstat$MSE), 2)
round(perf_lstat["lm_lstat", "MSE"], 2)


#*****************************************************************************
#*  TUTTE LE VARIABILI
#*****************************************************************************
# Usiamo tutti i 13 regressori. Il test set è sempre lo stesso di prima.

#=============================== Regressione lineare ==========================
lm_all_fit = lm_spec %>%
  fit(medv ~ ., data = boston_trainval[, vars_all])
summary(lm_all_fit$fit)

lm_all_pred = predict(lm_all_fit, new_data = boston_test[, vars_all])
lm_all_perf = perf(lm_all_pred$.pred, boston_test$medv)
lm_all_perf

#=================================== KNN ======================================
knn_reg_spec = nearest_neighbor(neighbors = K_reg, weight_func = "rectangular") %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_reg_fit = knn_reg_spec %>%
  fit(medv ~ ., data = boston_trainval[, vars_all])

knn_reg_pred = predict(knn_reg_fit, new_data = boston_test[, vars_all])
knn_reg_perf = perf(knn_reg_pred$.pred, boston_test$medv)
knn_reg_perf

#======================= Albero di regressione e pruning ======================
set.seed(1, sample.kind="Rejection")
tree_reg_spec = decision_tree() %>%
  set_engine('rpart') %>%
  set_mode('regression')

tree_reg_fit = tree_reg_spec %>%
  fit(medv ~ ., data = boston_trainval[, vars_all])

# Plot the tree model
rpart.plot(tree_reg_fit$fit, roundint = F)

# * Pruning con la tabella di complessita' e il criterio 1-SE : si sceglie l'albero piu' piccolo il cui errore di cross-validation
# (xerror) sta entro un errore standard dal minimo.
cptable_reg = data.frame(tree_reg_fit$fit$cptable)
cptable_reg

mincpindex = which.min(cptable_reg[,"xerror"])
LL = cptable_reg[mincpindex,"xerror"] - cptable_reg[mincpindex,"xstd"] # lower limit
UL = cptable_reg[mincpindex,"xerror"] + cptable_reg[mincpindex,"xstd"] # upper limit

# smallest tree with xerror between the two limits
best_index = which(cptable_reg[,"xerror"] > LL & cptable_reg[,"xerror"] < UL) %>% min()

#save the CP of our interest!
bestcp_reg = cptable_reg$CP[best_index]
bestcp_reg

# numero di split dell'albero scelto vs albero con xerror minimo
cptable_reg$nsplit[best_index]
cptable_reg$nsplit[mincpindex]

# Rifacciamo il fit con quel valore di CP
tree_reg_pruned_spec = decision_tree(cost_complexity = bestcp_reg) %>%
  set_engine('rpart') %>%
  set_mode('regression')

tree_reg_pruned_fit = tree_reg_pruned_spec %>%
  fit(medv ~ ., data = boston_trainval[, vars_all])

rpart.plot(x = tree_reg_pruned_fit$fit, roundint = F)

tree_reg_pred = predict(tree_reg_fit, new_data = boston_test[, vars_all])
tree_reg_pruned_pred = predict(tree_reg_pruned_fit, new_data = boston_test[, vars_all])

tree_reg_perf = perf(tree_reg_pred$.pred, boston_test$medv)
tree_reg_pruned_perf = perf(tree_reg_pruned_pred$.pred, boston_test$medv)

rbind(tree = tree_reg_perf,
      pruned = tree_reg_pruned_perf)

# Un albero piu' piccolo e' piu' semplice da interpretare, ma non e' detto che predica meglio: il test set dice quanto "costa" la semplicita'.

#================================== Bagging ===================================
# Il bagging e' un random forest in cui ad ogni split si considerano TUTTI i regressori (mtry = p = 13). Con importance = T stimiamo l'importanza delle variabili.
set.seed(1, sample.kind="Rejection")
bag_spec = rand_forest(mtry = length(vars_all) - 1, # all the regressors
                       trees = 500) %>%
  set_engine('randomForest', importance = T) %>% # to estimate predictors importance
  set_mode('regression')

bag_fit = bag_spec %>%
  fit(medv ~ ., data = boston_trainval[, vars_all])
bag_fit

# Mean of squared residuals e % Var explained sono calcolate sulle osservazioni out-of-bag, quindi senza usare il test set.
bag_pred = predict(bag_fit, new_data = boston_test[, vars_all])
bag_perf = perf(bag_pred$.pred, boston_test$medv)
bag_perf

varImpPlot(bag_fit$fit)
importance(bag_fit$fit)

#  Le variabili in cima al grafico sono quelle che, se escluse, fanno aumentare di piu' l'errore. Come cambia l'errore OOB con il numero di alberi:
set.seed(22, sample.kind="Rejection")
ntree_vec = seq(10, 500, by=20)
ooberrvec = c() 

for(i in 1:length(ntree_vec)){
  spec = rand_forest(mtry = length(vars_all) - 1,
                     trees = ntree_vec[i]) %>%   
    set_engine('randomForest') %>%
    set_mode('regression')
  model = spec %>% fit(medv ~ ., data = boston_trainval[, vars_all])
  ooberrvec[i] = model$fit$mse[ntree_vec[i]] #extract the OOB MSE
}

data.frame(ntree_vec, ooberrvec) %>%
  ggplot() +
  geom_line(aes(ntree_vec, ooberrvec))+
  xlab("Number of trees")+
  ylab("OOB MSE")

#  Dopo poche centinaia di alberi la curva si stabilizza: aggiungere alberi non fa overfitting, ma smette di aiutare. Con B=500 siamo tranquilli.

#============================ Gradient boosting ===============================
# Parametri di complessita': numero di alberi B, learning rate lambda (shrinkage), profondita' d (interaction.depth).
# Usiamo la cross-validation interna di gbm (5 fold) su training+validation.

set.seed(1, sample.kind="Rejection")
gbm_first = gbm(formula = medv ~ .,
                data = boston_trainval[, vars_all],
                distribution = "gaussian",
                n.trees = 5000,
                shrinkage = 0.1,
                interaction.depth = 1,
                cv.folds = 5)

min(gbm_first$cv.error)
which.min(gbm_first$cv.error)
gbm.perf(gbm_first)

hyper_grid = expand.grid(
  shrinkage = c(0.005, .01, .1),
  interaction.depth = c(1, 3, 5)
)
hyper_grid$minMSE = NA
hyper_grid$bestB = NA

for(i in 1:nrow(hyper_grid)){

  print(paste("Iteration n.", i))
  set.seed(1, sample.kind="Rejection")
  gbm_grid = gbm(formula = medv ~ .,
                 data = boston_trainval[, vars_all],
                 distribution = "gaussian",
                 n.trees = 5000,
                 shrinkage = hyper_grid$shrinkage[i],
                 interaction.depth = hyper_grid$interaction.depth[i],
                 cv.folds = 5)

  hyper_grid$minMSE[i] = min(gbm_grid$cv.error)
  hyper_grid$bestB[i]  = which.min(gbm_grid$cv.error)
}

hyper_grid %>%
  arrange(minMSE)

best_row = hyper_grid %>%
  arrange(minMSE) %>%
  slice(1)
best_row

# Legame inverso tra lambda e B ottimo: piu' piccolo e' il learning rate, piu' alberi servono per arrivare al minimo.

# --- Modello finale: parametri migliori e piu' alberi, B ottimo da gbm.perf
set.seed(1, sample.kind="Rejection")
gbm_final = gbm(formula = medv ~ .,
                data = boston_trainval[, vars_all],
                distribution = "gaussian",
                n.trees = 10000,
                shrinkage = best_row$shrinkage,
                interaction.depth = best_row$interaction.depth,
                cv.folds = 5)

bestB = gbm.perf(gbm_final)
bestB

pred_boo = predict(gbm_final,
                   boston_test[, vars_all],
                   n.trees = bestB)
boo_perf = perf(pred_boo, boston_test$medv)
boo_perf

summary(gbm_final)
# L'influenza relativa dice quanto ogni variabile contribuisce alla riduzione dell'errore quadratico (somma 100). Confrontala con
# l'importanza del bagging e con la matrice di correlazione iniziale.


#*****************************************************************************
#*  CONFRONTO FINALE (REGRESSIONE)
#*****************************************************************************

perf_all = rbind(perf_lstat,
                 lm_all = lm_all_perf,
                 knn_all = knn_reg_perf,
                 tree_all = tree_reg_perf,
                 pruned_all = tree_reg_pruned_perf,
                 bagging_all = bag_perf,
                 boosting_all = boo_perf)

results_reg = perf_all %>%
  rownames_to_column("model") %>%
  mutate(regressors = ifelse(grepl("lstat", model), "only lstat", "all variables")) %>%
  arrange(MSE)
results_reg

results_reg %>%
  ggplot() +
  geom_col(aes(x=reorder(model, -MSE), y=MSE, fill=regressors)) +
  coord_flip() +
  xlab("model") +
  ylab("test MSE")

# modello migliore e peggiore
results_reg$model[1]
round(results_reg$MSE[1], 2)
round(results_reg$COR[1], 3)
results_reg$model[nrow(results_reg)]
round(results_reg$MSE[nrow(results_reg)], 2)

# Osservati vs previsti sul test set per tre modelli rappresentativi: spline naturale (migliore con la sola lstat), bagging e boosting.
# I punti vicini alla bisettrice rossa sono previsioni quasi esatte.
data.frame(obs = boston_test$medv,
           natural_lstat = ns_pred,
           bagging_all = bag_pred$.pred,
           boosting_all = pred_boo) %>%
  pivot_longer(cols = natural_lstat:boosting_all,
               names_to = "model",
               values_to = "pred") %>%
  ggplot(aes(obs, pred)) +
  geom_point() +
  geom_abline(col = "red") +
  facet_wrap(~model)


#*****************************************************************************
#*  PARTE B: PENGUINS, CLASSIFICAZIONE
#*****************************************************************************
# Obiettivo: riconoscere il SESSO (sex, binario) di un pinguino dalla specie e da quattro misure corporee. Confronto di modelli, soglie e ROC.

#=============================== Dati ed esplorazione =========================
glimpse(penguins)
# Checking missing values
colSums(is.na(penguins))

penguin = penguins %>%
  dplyr::select(-island) %>%
  drop_na()

dim(penguin)
table(penguin$sex)


penguin %>%
  ggplot(aes(bill_length_mm, bill_depth_mm, col=species))+
  geom_point()+
  facet_wrap(~sex)

penguin %>%
  ggplot() +
  geom_boxplot(aes(x=sex, y=body_mass_g, fill=sex)) +
  facet_wrap(~species)

# Dentro ogni specie i maschi sono piu' pesanti, ma tra specie diverse i pesi si sovrappongono: il peso e' informativo sul sesso SOLO una volta
# fissata la specie. Per questo species resta tra i regressori.

ggpairs(penguin)
# Le variabili numeriche sono correlate (le misure crescono con la taglia), quindi l'ipotesi del Naive Bayes e' violata. Lo proviamo comunque.

#============== Training, validation e test set (metodo 2) ====================
# 60% training, 20% validation, 20% test.
penguin = penguin %>%
  mutate(id = row_number()) #create a new column with the row numbers

set.seed(1, sample.kind="Rejection")
penguin_training = penguin %>%
  slice_sample(prop=0.6)
penguin_rest = anti_join(x=penguin, y=penguin_training, by="id")

# the remaining observations are split in two halves: validation and test
penguin_validation = penguin_rest %>%
  slice_sample(prop=0.5)
penguin_test = anti_join(x=penguin_rest, y=penguin_validation, by="id")

# the id column is not needed anymore
penguin = penguin %>% dplyr::select(-id)
penguin_training = penguin_training %>% dplyr::select(-id)
penguin_validation = penguin_validation %>% dplyr::select(-id)
penguin_test = penguin_test %>% dplyr::select(-id)

c(training = nrow(penguin_training),
  validation = nrow(penguin_validation),
  test = nrow(penguin_test))
# Con un test set cosi' piccolo le metriche sono rumorose: differenze piccole tra modelli non vanno prese troppo sul serio.

#=============================== Indici di performance ========================
levels(penguin$sex)

indexes = metric_set(accuracy, sens, yardstick::spec)

# function computing the performance indexes from observed and predicted classes
class_indexes = function(obs, pred){
  data.frame(obs = obs, pred = pred) %>%
    indexes(truth = obs, estimate = pred)
}

#============================ Tuning di K per il KNN ==========================
maxK = 50
errorrate = c()

for (i in 1:maxK) {
  # KNN classification
  knn_spec = nearest_neighbor(weight_func = "rectangular", neighbors = i) %>%
    set_engine("kknn") %>%
    set_mode("classification")

  knn_fit = knn_spec %>%
    fit(sex ~ ., data = penguin_training)

  # Make predictions
  pred2 = predict(knn_fit, penguin_validation, type = "class")

  # Validation error rate
  errorrate[i] = mean(pred2$.pred_class != penguin_validation$sex)
}

data.frame(k=1:maxK, validER = errorrate) %>%
  ggplot() +
  geom_line(aes(x=k, y=validER))

which(errorrate == min(errorrate)) # all the K with the lowest error rate

K_class = max(which(errorrate == min(errorrate)))
K_class

# Riuniamo training e validation: tutti i modelli finali sono stimati su questo insieme e valutati sullo stesso test set.
penguin_trainval = rbind(penguin_validation, penguin_training)
dim(penguin_trainval)

#============================ Regressione logistica ===========================
logreg_spec = logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Fit the logistic regression model
logreg_fit = logreg_spec %>%
  fit(sex ~ ., data = penguin_trainval)
summary(logreg_fit$fit)

# odds ratios: multiplicative change of the odds of being male for a one-unit increase
exp(logreg_fit$fit$coefficients)
#  glm modella la probabilita' del secondo livello, cioe' male. Odds ratio > 1: l'aumento del regressore fa crescere le odds di essere maschio.
# Per i livelli di species il confronto e' con la specie di riferimento, Adelie.

logreg_pred = logreg_fit %>%
  predict(new_data = penguin_test, type = "prob") %>%
  bind_cols(logreg_fit %>%
              predict(new_data = penguin_test, type = "class"))
head(logreg_pred)

table(pred = logreg_pred$.pred_class, obs = penguin_test$sex)

index_lg = class_indexes(penguin_test$sex, logreg_pred$.pred_class)
index_lg

# Cosa succede cambiando la soglia (0.8 invece di 0.5) ?
pred_08 = factor(ifelse(logreg_pred$.pred_male > 0.8, "male", "female"),
                 levels = levels(penguin_test$sex))
table(pred = pred_08, obs = penguin_test$sex)
class_indexes(penguin_test$sex, pred_08)
#  Alzando la soglia diciamo "maschio" piu' raramente: la sensitivity puo' solo aumentare e la specificity solo diminuire. La soglia giusta dipende
# dal contesto.

# ROC curve for Logistic Regression
roc_logreg = roc(penguin_test$sex, logreg_pred$.pred_male)
auc(roc_logreg)

#=================================== LDA ======================================

lda_spec = discrim_linear() %>%
  set_engine("MASS") %>%
  set_mode("classification")

lda_fit = lda_spec %>%
  fit(sex ~ ., data = penguin_trainval)
lda_fit

lda_pred = lda_fit %>%
  predict(new_data = penguin_test, type = 'prob') %>%
  bind_cols(lda_fit %>%
              predict(new_data = penguin_test, type = 'class'))

table(pred = lda_pred$.pred_class, obs = penguin_test$sex)

index_lda = class_indexes(penguin_test$sex, lda_pred$.pred_class)
index_lda

roclda = roc(penguin_test$sex, lda_pred$.pred_male)
auc(roclda)

#=================================== QDA ======================================
qda_spec = discrim_quad() %>%
  set_engine("MASS") %>%
  set_mode("classification")

qda_fit = qda_spec %>%
  fit(sex ~ ., data = penguin_trainval)

qda_pred = predict(qda_fit, new_data = penguin_test, type = 'prob') %>%
  bind_cols(predict(qda_fit, new_data = penguin_test, type = 'class'))

table(pred = qda_pred$.pred_class, obs = penguin_test$sex)

index_qda = class_indexes(penguin_test$sex, qda_pred$.pred_class)
index_qda

rocqda = roc(penguin_test$sex, qda_pred$.pred_male)
auc(rocqda)

#================================ Naive Bayes =================================
classifier = naiveBayes(sex ~ ., data = penguin_trainval)
classifier$apriori # prior distribution

na_pred = predict(classifier,
                  newdata = dplyr::select(penguin_test, -sex))
na_prob = predict(classifier,
                  newdata = dplyr::select(penguin_test, -sex), type = "raw")
head(na_prob)

table(pred = na_pred, obs = penguin_test$sex)

index_na = class_indexes(penguin_test$sex, na_pred)
index_na

rocnb = roc(penguin_test$sex, na_prob[, "male"])
auc(rocnb)

#=================================== KNN ======================================
# Con il valore di K scelto sul validation set
knn_spec = nearest_neighbor(neighbors = K_class, weight_func = "rectangular") %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_fit = knn_spec %>%
  fit(sex ~ ., data = penguin_trainval)

knn_pred = predict(knn_fit, penguin_test, type = "class") %>%
  bind_cols(predict(knn_fit, penguin_test, type = "prob"))

table(pred = knn_pred$.pred_class, obs = penguin_test$sex)

index_knn = class_indexes(penguin_test$sex, knn_pred$.pred_class)
index_knn

rocknn = roc(penguin_test$sex, knn_pred$.pred_male)
auc(rocknn)

#================== Albero di classificazione e pruning =======================
set.seed(1, sample.kind="Rejection")
tree_fit = decision_tree() %>%
  set_engine('rpart') %>%
  set_mode('classification') %>%
  fit(sex ~ . , data = penguin_trainval)

sum(tree_fit$fit$frame$var=="<leaf>") # number of leaves
rpart.plot(x = tree_fit$fit, roundint=FALSE)


cptable = data.frame(tree_fit$fit$cptable)
cptable

mincpindex = which.min(cptable[,"xerror"])
LL = cptable[mincpindex,"xerror"] - cptable[mincpindex,"xstd"]
UL = cptable[mincpindex,"xerror"] + cptable[mincpindex,"xstd"]
best_index = which(cptable[,"xerror"] > LL & cptable[,"xerror"] < UL) %>% min()
bestcp = cptable$CP[best_index]
bestcp

set.seed(1, sample.kind="Rejection")
tree_pruned_fit = decision_tree(cost_complexity = bestcp) %>%
  set_engine('rpart') %>%
  set_mode('classification') %>%
  fit(sex ~ . , data = penguin_trainval)

rpart.plot(x = tree_pruned_fit$fit, roundint=FALSE)

tree_pred = predict(tree_pruned_fit, new_data = penguin_test, type = "prob") %>%
  bind_cols(predict(tree_pruned_fit, new_data = penguin_test, type = "class"))

table(pred = tree_pred$.pred_class, obs = penguin_test$sex)

index_tree = class_indexes(penguin_test$sex, tree_pred$.pred_class)
index_tree

# accuracy of the unpruned tree, for comparison
tree_full_class = predict(tree_fit, new_data = penguin_test, type = "class")
mean(tree_full_class$.pred_class == penguin_test$sex)

roctree = roc(penguin_test$sex, tree_pred$.pred_male)
auc(roctree)
# Nel confronto finale usiamo l'albero potato. Le probabilita' sono le proporzioni di classe nelle foglie, quindi la ROC ha pochi gradini.

#=============================== Random forest ================================

set.seed(1, sample.kind="Rejection")
rf_fit = rand_forest(mtry = floor(sqrt(ncol(penguin_trainval) - 1)), # sqrt(p)
                     trees = 1000) %>%
  set_engine('randomForest', importance = T) %>%
  set_mode('classification') %>%
  fit(sex ~ ., data = penguin_trainval)
print(rf_fit)

varImpPlot(rf_fit$fit)

rf_pred = predict(rf_fit, new_data = penguin_test, type = "prob") %>%
  bind_cols(predict(rf_fit, new_data = penguin_test, type = "class"))

table(pred = rf_pred$.pred_class, obs = penguin_test$sex)

index_rf = class_indexes(penguin_test$sex, rf_pred$.pred_class)
index_rf

rocrf = roc(penguin_test$sex, rf_pred$.pred_male)
auc(rocrf)


roc_list = list(logreg = roc_logreg,
                lda = roclda,
                qda = rocqda,
                nb = rocnb,
                knn = rocknn,
                tree = roctree,
                rf = rocrf)

ggroc(roc_list)

index_all = bind_rows(logreg = index_lg,
                      lda = index_lda,
                      qda = index_qda,
                      nb = index_na,
                      knn = index_knn,
                      tree = index_tree,
                      rf = index_rf,
                      .id = "model") %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

auc_all = data.frame(model = names(roc_list),
                     auc = sapply(roc_list, function(x) as.numeric(auc(x))),
                     row.names = NULL)

results_class = index_all %>%
  left_join(auc_all, by = "model") %>%
  arrange(desc(auc))
results_class

# modello con AUC piu' alta e piu' bassa
results_class$model[1]
round(results_class$auc[1], 3)
results_class$model[nrow(results_class)]
round(results_class$auc[nrow(results_class)], 3)



#*****************************************************************************
#*  PARTE C: PENGUINS, CLUSTERING E PCA
#*****************************************************************************
# Usiamo solo le quattro misure corporee. Le specie reali servono SOLO alla fine per valutare i cluster.

#================================ Standardizzazione ===========================
# media e varianza delle variabili
penguin %>%
  select_if(is.numeric) %>%
  summarise_all(mean)

penguin %>%
  select_if(is.numeric) %>%
  summarise_all(stats::var)

# Varianze su scale diverse (il peso in grammi domina): standardizziamo con scale, altrimenti la distanza euclidea dipenderebbe quasi solo da body_mass_g.
penguin_std = penguin %>%
  select_if(is.numeric) %>%
  scale()

apply(penguin_std, 2, mean)
apply(penguin_std, 2, stats::var)

# Matrice delle distanze con fviz_dist: piu' blu = maggiore dissimilarita'
distmat = dist(penguin_std)
fviz_dist(distmat, show_labels = FALSE)

#============================= Clustering gerarchico ==========================
clust_avg = hclust(distmat, method = "average")

plot(clust_avg,
     labels = FALSE, 
     hang = -1)
rect.hclust(clust_avg,
            k = 3, #how many clusters
            border = c("red","blue","green"))

clust_avgcut = cutree(clust_avg, k = 3)
table(clust_avgcut)

# Con il legame "complete" otteniamo un albero diverso
clust_comp = hclust(distmat, method = "complete")

plot(clust_comp,
     labels = FALSE,
     hang = -1)
rect.hclust(clust_comp,
            k = 3,
            border = c("red","blue","green"))

clust_compcut = cutree(clust_comp, k = 3)
table(clust_compcut)

#================================== K-means ===================================
# Numero di cluster: somma dei quadrati entro i gruppi (wss) al variare di K, cercando un gomito 
fviz_nbclust(penguin_std,
             kmeans,
             method = "wss")

# Per confrontare con le tre specie reali usiamo K=3.
set.seed(1, sample.kind="Rejection")
clust_km = kmeans(penguin_std,
                  centers = 3, #n. of clusters
                  nstart = 10) #run the alg. 10 times
clust_km

autoplot(clust_km,
         data = penguin_std,
         frame = TRUE)

#=================================== PCA ======================================
#  Vista a due dimensioni dei dati e lettura delle direzioni principali 
pca = penguin %>%
  select_if(is.numeric) %>%
  prcomp(center = T, #zero-mean data
         scale = T) #unit variance data

pca$rotation

pca_tab = summary(pca)
pca_tab
screeplot(pca)

data.frame(PC = 1:4,
           cumvar = pca_tab$importance[3,]) %>%
  ggplot(aes(PC, cumvar)) +
  geom_point() +
  geom_line()

# varianza spiegata da PC1 e dalle prime due componenti (%)
round(100*pca_tab$importance[2,1], 1)
round(100*pca_tab$importance[3,2], 1)

# Correlazioni tra punteggi e variabili originali
data.frame(penguin %>% select_if(is.numeric), PC1=pca$x[,1]) %>%
  cor() %>%
  ggcorrplot(lab = TRUE,
             lab_size=3)

data.frame(penguin %>% select_if(is.numeric), PC2=pca$x[,2]) %>%
  cor() %>%
  ggcorrplot(lab = TRUE,
             lab_size=3)

# Se peso, pinna e lunghezza del becco hanno lo stesso segno su PC1, questa e' una componente di TAGLIA complessiva. Per PC2 guarda i pesi piu' grandi
# in valore assoluto nella tabella dei loadings. Biplot: loadings (frecce) e punteggi (pinguini), colore = specie, forma = sesso
autoplot(pca,
         data = as.data.frame(penguin),
         colour = "species",
         shape = "sex",
         loadings = TRUE,
         loadings.label = TRUE)

#======================== Cluster contro specie reali =========================
# Confrontiamo i cluster con le specie e misuriamo la PUREZZA: quota di pinguini che appartiene alla specie piu' frequente del proprio cluster
#  (1 = cluster perfettamente coincidenti con le specie).
table(cluster = clust_avgcut, species = penguin$species)
table(cluster = clust_compcut, species = penguin$species)
table(cluster = clust_km$cluster, species = penguin$species)

# share of penguins belonging to the most frequent species of their cluster
purity = function(cl, truth){
  tab = table(cl, truth)
  return(sum(apply(tab, 1, max)) / sum(tab))
}

purity_all = data.frame(method = c("hclust average", "hclust complete", "k-means"),
                        purity = c(purity(clust_avgcut, penguin$species),
                                   purity(clust_compcut, penguin$species),
                                   purity(clust_km$cluster, penguin$species)))
purity_all

# metodo che ricostruisce meglio le specie
purity_all$method[which.max(purity_all$purity)]
round(max(purity_all$purity), 3)

# Cluster e specie sulle prime due componenti: colore = specie reale, forma = cluster di K-means
data.frame(pca$x[,1:2],
           species = penguin$species,
           cluster = factor(clust_km$cluster)) %>%
  ggplot(aes(PC1, PC2, col = species, shape = cluster)) +
  geom_point(size = 2)

#  I pinguini mal raggruppati sono quelli in cui forma e colore non vanno d'accordo. Le specie si sovrappongono in alcune zone: nessun metodo non
#  supervisionato puo' fare meglio di quello che le variabili permettono.






