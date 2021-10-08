#Necesita para correr en Google Cloud
#96 GB de memoria RAM
#300 GB de espacio en el disco local
#8 vCPU

#clase_binaria2   1={BAJA+2,BAJA+1}    0={CONTINUA}
#Entrena en a union de ONCE meses de [202001, 202011]
#No usa variables historicas

#Optimizacion Bayesiana de hiperparametros de  lightgbm
#usa el interminable  5-fold cross validation
#funciona automaticamente con EXPERIMENTOS
#va generando incrementalmente salidas para kaggle

# WARNING  usted debe cambiar este script si lo corre en su propio Linux

#limpio la memoria
#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")
require("yaml")

require("lightgbm")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")


#para poder usarlo en la PC y en la nube sin tener que cambiar la ruta
#cambiar aqui las rutas en su maquina
switch ( Sys.info()[['sysname']],
         Windows = { directory.root  <-  "M:\\" },   #Windows
         Darwin  = { directory.root  <-  "~/dm/" },  #Apple MAC
         Linux   = { directory.root  <-  "~/buckets/b1/" } #Google Cloud
)
#defino la carpeta donde trabajo
setwd( directory.root )



kexperimento  <- NA   #NA si se corre la primera vez, un valor concreto si es para continuar procesando

kscript         <- "lgb2-b2-lag12-fe-elimvarext-coc"

karch_dataset    <- "./datasetsOri/paquete_premium.csv.gz"
kmes_apply       <- 202101  #El mes donde debo aplicar el modelo
kmes_train_hasta <- 202011  #Obvimente, solo puedo entrenar hasta 202011

kmes_train_desde <- 202001  #Entreno desde Enero-2020

kcanaritos  <-  100
kBO_iter    <-  50   #cantidad de iteraciones de la Optimizacion Bayesiana

#Aqui se cargan los hiperparametros
hs <- makeParamSet( 
  makeNumericParam("learning_rate",    lower=    0.015 , upper=    0.06),#CHECK
  makeNumericParam("feature_fraction", lower=    0.2  , upper=    0.4), #CHECK 
  makeIntegerParam("min_data_in_leaf", lower= 1000L   , upper= 4000L), #CHECK
  makeIntegerParam("num_leaves",       lower=  500L   , upper= 1400L), #CHECK
  makeNumericParam("prob_corte",       lower=    0.020, upper=    0.05)#CHECK
)

campos_malos  <- c("mpasivos_margen","mcuentas_saldo","mautoservicio", "cpagomiscuentas", "mpagomiscuentas", "mtarjeta_visa_descuentos", "ctrx_quarter", "Master_mfinanciacion_limite", "Master_mconsumospesos", "Master_fultimo_cierre","Master_mpagominimo","Visa_mfinanciacion_limite","Visa_msaldopesos","Visa_mconsumospesos", "Visa_fultimo_cierre", "Visa_mpagospesos" )  

#aqui se deben cargar todos los campos culpables del Data Drifting
#Me fije cuando aparecian con datadrifting+por encima de canarios

campos_malos2 <- c("ccomisiones_otras_lag2/cproductos_delta1","ccomisiones_otras_lag2/cproductos_delta2","ccomisiones_otras_lag2/ctarjeta_debito_transacciones","ccomisiones_otras_lag2/ctarjeta_debito_transacciones_lag1","ccomisiones_otras_lag2/ctarjeta_visa_transacciones","ccomisiones_otras_lag2/ctarjeta_visa_transacciones_lag1","ccomisiones_otras_lag2/ctarjeta_visa_transacciones_lag2","ccomisiones_otras/ccaja_ahorro","ccomisiones_otras/ccomisiones_otras_lag2","ccomisiones_otras/cextraccion_autoservicio","ccomisiones_otras/cpayroll_trx","ccomisiones_otras/cpayroll_trx_lag1","ccomisiones_otras/cpayroll_trx_lag2","ccomisiones_otras/cprestamos_personales","ccomisiones_otras/cproductos","ccomisiones_otras/cproductos_delta1","ccomisiones_otras/cproductos_delta2","ccomisiones_otras/ctarjeta_debito_transacciones","ccomisiones_otras/ctarjeta_debito_transacciones_lag1","ccomisiones_otras/ctarjeta_visa_transacciones","ccomisiones_otras/ctarjeta_visa_transacciones_lag1","ccomisiones_otras/ctarjeta_visa_transacciones_lag2")

campos_malos2bisbisbis <- c("cextraccion_autoservicio","cextraccion_autoservicio_tend1","cextraccion_autoservicio/ccaja_ahorro","cextraccion_autoservicio/ccomisiones_otras","cextraccion_autoservicio/ccomisiones_otras_lag2","cextraccion_autoservicio/cpayroll_trx","cextraccion_autoservicio/cpayroll_trx_lag1","cextraccion_autoservicio/cpayroll_trx_lag2","cextraccion_autoservicio/cprestamos_personales","cextraccion_autoservicio/cproductos","cextraccion_autoservicio/cproductos_delta1","cextraccion_autoservicio/cproductos_delta2","cextraccion_autoservicio/ctarjeta_debito_transacciones","cextraccion_autoservicio/ctarjeta_debito_transacciones_lag1","cextraccion_autoservicio/ctarjeta_visa_transacciones_lag1","cextraccion_autoservicio/ctarjeta_visa_transacciones_lag2","chomebanking_transacciones_lag1","chomebanking_transacciones_lag2","cliente_antiguedad","cliente_antiguedad_lag1","cliente_antiguedad_lag2","cpagodeservicios","cpayroll_trx","cpayroll_trx_lag1","cpayroll_trx_lag1/ccaja_ahorro","cpayroll_trx_lag1/ccomisiones_otras","cpayroll_trx_lag1/ccomisiones_otras_lag2","cpayroll_trx_lag1/cextraccion_autoservicio","cpayroll_trx_lag1/cpayroll_trx","cpayroll_trx_lag1/cpayroll_trx_lag2","cpayroll_trx_lag1/cprestamos_personales","cpayroll_trx_lag1/cproductos","cpayroll_trx_lag1/cproductos_delta1","cpayroll_trx_lag1/cproductos_delta2","cpayroll_trx_lag1/ctarjeta_debito_transacciones_lag1","cpayroll_trx_lag1/ctarjeta_visa_transacciones","cpayroll_trx_lag1/ctarjeta_visa_transacciones_lag1","cpayroll_trx_lag1/ctarjeta_visa_transacciones_lag2","cpayroll_trx_lag2","cpayroll_trx_lag2/ccaja_ahorro","cpayroll_trx_lag2/ccomisiones_otras","cpayroll_trx_lag2/ccomisiones_otras_lag2","cpayroll_trx_lag2/cextraccion_autoservicio","cpayroll_trx_lag2/cpayroll_trx","cpayroll_trx_lag2/cpayroll_trx_lag1","cpayroll_trx_lag2/cprestamos_personales","cpayroll_trx_lag2/cproductos","cpayroll_trx_lag2/cproductos_delta1","cpayroll_trx_lag2/cproductos_delta2","cpayroll_trx_lag2/ctarjeta_debito_transacciones","cpayroll_trx_lag2/ctarjeta_debito_transacciones_lag1","cpayroll_trx_lag2/ctarjeta_visa_transacciones","cpayroll_trx_lag2/ctarjeta_visa_transacciones_lag1","cpayroll_trx_lag2/ctarjeta_visa_transacciones_lag2","cpayroll_trx_tend1","cpayroll_trx/ccomisiones_otras","cpayroll_trx/ccomisiones_otras_lag2","cpayroll_trx/cextraccion_autoservicio","cpayroll_trx/cpayroll_trx_lag1","cpayroll_trx/cpayroll_trx_lag2","cpayroll_trx/cproductos","cpayroll_trx/cproductos_delta1","cpayroll_trx/cproductos_delta2","cpayroll_trx/ctarjeta_debito_transacciones_lag1","cpayroll_trx/ctarjeta_visa_transacciones_lag1","cpayroll_trx/ctarjeta_visa_transacciones_lag2","cprestamos_personales","cprestamos_personales_tend1","cprestamos_personales/ccaja_ahorro","cprestamos_personales/ccomisiones_otras","cprestamos_personales/ccomisiones_otras_lag2","cprestamos_personales/cextraccion_autoservicio","cprestamos_personales/cpayroll_trx","cprestamos_personales/cpayroll_trx_lag1","cprestamos_personales/cpayroll_trx_lag2","cprestamos_personales/cproductos","cprestamos_personales/cproductos_delta1","cprestamos_personales/cproductos_delta2","cprestamos_personales/ctarjeta_debito_transacciones","cprestamos_personales/ctarjeta_debito_transacciones_lag1","cprestamos_personales/ctarjeta_visa_transacciones","cprestamos_personales/ctarjeta_visa_transacciones_lag1","cprestamos_personales/ctarjeta_visa_transacciones_lag2","cproductos_delta1","cproductos_delta1/ccaja_ahorro","cproductos_delta1/ccomisiones_otras","cproductos_delta1/ccomisiones_otras_lag2","cproductos_delta1/cextraccion_autoservicio","cproductos_delta1/cpayroll_trx","cproductos_delta1/cpayroll_trx_lag1","cproductos_delta1/cpayroll_trx_lag2","cproductos_delta1/cprestamos_personales","cproductos_delta1/cproductos","cproductos_delta1/cproductos_delta2")

campos_malosbisbis <- c("ccaja_ahorro","ccaja_ahorro/ccomisiones_otras","ccaja_ahorro/ccomisiones_otras_lag2","ccaja_ahorro/cextraccion_autoservicio","ccaja_ahorro/cpayroll_trx_lag1","ccaja_ahorro/cpayroll_trx_lag2","ccaja_ahorro/cprestamos_personales","ccaja_ahorro/cproductos_delta1","ccaja_ahorro/cproductos_delta2","ccaja_ahorro/ctarjeta_debito_transacciones","ccaja_ahorro/ctarjeta_debito_transacciones_lag1","ccaja_ahorro/ctarjeta_visa_transacciones_lag1","ccaja_ahorro/ctarjeta_visa_transacciones_lag2","ccajeros_propios_descuentos","ccajeros_propios_descuentos_lag1","ccheques_emitidos_rechazados_delta2","ccomisiones_otras","ccomisiones_otras_lag2","ccomisiones_otras_lag2/ccaja_ahorro","ccomisiones_otras_lag2/ccomisiones_otras","ccomisiones_otras_lag2/cextraccion_autoservicio","ccomisiones_otras_lag2/cpayroll_trx","ccomisiones_otras_lag2/cpayroll_trx_lag1","ccomisiones_otras_lag2/cpayroll_trx_lag2","ccomisiones_otras_lag2/cprestamos_personales","ccomisiones_otras_lag2/cproductos")

campos_malos2bis <- c("cproductos_delta1/ctarjeta_debito_transacciones","cproductos_delta1/ctarjeta_debito_transacciones_lag1","cproductos_delta1/ctarjeta_visa_transacciones","cproductos_delta1/ctarjeta_visa_transacciones_lag1","cproductos_delta1/ctarjeta_visa_transacciones_lag2","cproductos_delta2","cproductos_delta2/ccaja_ahorro","cproductos_delta2/ccomisiones_otras","cproductos_delta2/ccomisiones_otras_lag2","cproductos_delta2/cextraccion_autoservicio","cproductos_delta2/cpayroll_trx_lag1","cproductos_delta2/cpayroll_trx_lag2","cproductos_delta2/cproductos","cproductos_delta2/cproductos_delta1")

campos_malos3 <- c("mactivos_margen_lag2/cpayroll_trx","mactivos_margen_lag2/cpayroll_trx_lag1","mactivos_margen_lag2/cpayroll_trx_lag2","mactivos_margen_lag2/cprestamos_personales","mactivos_margen_lag2/cproductos","mactivos_margen_lag2/cproductos_delta1","mactivos_margen_lag2/cproductos_delta2","mactivos_margen_lag2/ctarjeta_debito_transacciones","mactivos_margen_lag2/ctarjeta_debito_transacciones_lag1","mactivos_margen_lag2/ctarjeta_visa_transacciones","mactivos_margen_lag2/ctarjeta_visa_transacciones_lag1","mactivos_margen_lag2/ctarjeta_visa_transacciones_lag2","mactivos_margen/ccomisiones_otras","mactivos_margen/ccomisiones_otras_lag2","mactivos_margen/cextraccion_autoservicio","mactivos_margen/cpayroll_trx","mactivos_margen/cpayroll_trx_lag1","mactivos_margen/cpayroll_trx_lag2","mactivos_margen/cprestamos_personales","mactivos_margen/cproductos_delta1","mactivos_margen/cproductos_delta2","mactivos_margen/ctarjeta_debito_transacciones","mactivos_margen/ctarjeta_debito_transacciones_lag1","mactivos_margen/ctarjeta_visa_transacciones","Master_fechaalta","Master_fechaalta_lag1","Master_fechaalta_lag2","Master_fechaalta_lag2/ccaja_ahorro","Master_fechaalta_lag2/ccomisiones_otras_lag2","Master_fechaalta_lag2/cextraccion_autoservicio","Master_fechaalta_lag2/cpayroll_trx","Master_fechaalta_lag2/cpayroll_trx_lag1","Master_fechaalta_lag2/cpayroll_trx_lag2","Master_fechaalta_lag2/cprestamos_personales","Master_fechaalta_lag2/cproductos_delta1","Master_fechaalta_lag2/cproductos_delta2","Master_fechaalta_lag2/ctarjeta_debito_transacciones","Master_fechaalta_lag2/ctarjeta_debito_transacciones_lag1","Master_fechaalta_lag2/ctarjeta_visa_transacciones","Master_fechaalta_lag2/ctarjeta_visa_transacciones_lag1","Master_fechaalta_tend1","Master_Fvencimiento_lag1","Master_mlimitecompra","Master_mlimitecompra_lag1","Master_mlimitecompra_lag2","Master_mlimitecompra_lag2/ccaja_ahorro","Master_mlimitecompra_lag2/ccomisiones_otras","Master_mlimitecompra_lag2/ccomisiones_otras_lag2","Master_mlimitecompra_lag2/cextraccion_autoservicio","Master_mlimitecompra_lag2/cpayroll_trx","Master_mlimitecompra_lag2/cpayroll_trx_lag1","Master_mlimitecompra_lag2/cpayroll_trx_lag2","Master_mlimitecompra_lag2/cprestamos_personales","Master_mlimitecompra_lag2/cproductos","Master_mlimitecompra_lag2/cproductos_delta1","Master_mlimitecompra_lag2/cproductos_delta2","Master_mlimitecompra_lag2/ctarjeta_debito_transacciones","Master_mlimitecompra_lag2/ctarjeta_debito_transacciones_lag1","Master_mlimitecompra_lag2/ctarjeta_visa_transacciones","Master_mlimitecompra_lag2/ctarjeta_visa_transacciones_lag1","Master_mlimitecompra_lag2/ctarjeta_visa_transacciones_lag2","mcaja_ahorro_dolares","mcaja_ahorro_dolares_lag1","mcaja_ahorro_dolares_lag2","mcaja_ahorro_dolares/ccaja_ahorro","mcaja_ahorro_dolares/ccomisiones_otras","mcaja_ahorro_dolares/ccomisiones_otras_lag2","mcaja_ahorro_dolares/cextraccion_autoservicio","mcaja_ahorro_dolares/cpayroll_trx","mcaja_ahorro_dolares/cpayroll_trx_lag1","mcaja_ahorro_dolares/cpayroll_trx_lag2","mcaja_ahorro_dolares/cprestamos_personales","mcaja_ahorro_dolares/cproductos","mcaja_ahorro_dolares/cproductos_delta1","mcaja_ahorro_dolares/cproductos_delta2","mcaja_ahorro_dolares/ctarjeta_debito_transacciones","mcaja_ahorro_dolares/ctarjeta_debito_transacciones_lag1","mcaja_ahorro_dolares/ctarjeta_visa_transacciones","mcaja_ahorro_dolares/ctarjeta_visa_transacciones_lag1","mcaja_ahorro_dolares/ctarjeta_visa_transacciones_lag2","mcheques_depositados_rechazados","mcheques_emitidos_rechazados_delta1","mcheques_emitidos_rechazados_lag2","mcomisiones","mcomisiones_delta2","mcomisiones_lag1","mcomisiones_lag2","mcomisiones_lag2/ccaja_ahorro","mcomisiones_lag2/ccomisiones_otras_lag2","mcomisiones_lag2/cextraccion_autoservicio")

campos_malos3bis<-c("mcomisiones_lag2/cpayroll_trx","mcomisiones_lag2/cpayroll_trx_lag1","mcomisiones_lag2/cpayroll_trx_lag2","mcomisiones_lag2/cprestamos_personales","mcomisiones_lag2/cproductos","mcomisiones_lag2/cproductos_delta1","mcomisiones_lag2/cproductos_delta2","mcomisiones_lag2/ctarjeta_debito_transacciones","mcomisiones_lag2/ctarjeta_debito_transacciones_lag1","mcomisiones_lag2/ctarjeta_visa_transacciones","mcomisiones_lag2/ctarjeta_visa_transacciones_lag1","mcomisiones_lag2/ctarjeta_visa_transacciones_lag2","mcomisiones_mantenimiento_lag2","mcomisiones_otras","mcomisiones_otras_delta2","mcomisiones_otras_lag1","mcomisiones_otras_lag2","mcomisiones_otras/ccaja_ahorro","mcomisiones_otras/ccomisiones_otras","mcomisiones_otras/ccomisiones_otras_lag2","mcomisiones_otras/cextraccion_autoservicio","mcomisiones_otras/cpayroll_trx","mcomisiones_otras/cpayroll_trx_lag1","mcomisiones_otras/cpayroll_trx_lag2","mcomisiones_otras/cprestamos_personales","mcomisiones_otras/cproductos_delta1","mcomisiones_otras/cproductos_delta2","mcomisiones_otras/ctarjeta_debito_transacciones","mcomisiones_otras/ctarjeta_debito_transacciones_lag1","mcomisiones_otras/ctarjeta_visa_transacciones","mcomisiones_otras/ctarjeta_visa_transacciones_lag1","mcomisiones_otras/ctarjeta_visa_transacciones_lag2","mcuenta_corriente","mcuenta_corriente_delta1","mcuenta_corriente_lag1","mcuenta_corriente_lag1/ccaja_ahorro","mcuenta_corriente_lag1/ccomisiones_otras","mcuenta_corriente_lag1/ccomisiones_otras_lag2","mcuenta_corriente_lag1/cextraccion_autoservicio","mcuenta_corriente_lag1/cpayroll_trx","mcuenta_corriente_lag1/cpayroll_trx_lag1","mcuenta_corriente_lag1/cpayroll_trx_lag2","mcuenta_corriente_lag1/cprestamos_personales","mcuenta_corriente_lag1/cproductos","mcuenta_corriente_lag1/cproductos_delta1","mcuenta_corriente_lag1/cproductos_delta2","mcuenta_corriente_lag1/ctarjeta_debito_transacciones","mcuenta_corriente_lag1/ctarjeta_debito_transacciones_lag1","mcuenta_corriente_lag1/ctarjeta_visa_transacciones","mcuenta_corriente_lag1/ctarjeta_visa_transacciones_lag1","mcuenta_corriente_lag1/ctarjeta_visa_transacciones_lag2","mcuenta_corriente_lag2","mcuenta_corriente/ccaja_ahorro","mcuenta_corriente/cpayroll_trx_lag1","mcuenta_corriente/cpayroll_trx_lag2","mcuenta_corriente/cproductos","mcuenta_corriente/cproductos_delta1","mcuenta_corriente/cproductos_delta2","mcuenta_corriente/ctarjeta_debito_transacciones","mcuenta_corriente/ctarjeta_visa_transacciones","mcuenta_corriente/ctarjeta_visa_transacciones_lag1","mcuenta_corriente/ctarjeta_visa_transacciones_lag2","mdescubierto_preacordado","mdescubierto_preacordado_lag1","mdescubierto_preacordado/ccomisiones_otras","ctarjeta_visa_transacciones/ctarjeta_debito_transacciones_lag1","ctarjeta_visa_transacciones/ctarjeta_visa_transacciones_lag1","ctarjeta_visa_transacciones/ctarjeta_visa_transacciones_lag2","mactivos_margen","mactivos_margen_lag1","mactivos_margen_lag1/ccaja_ahorro","mactivos_margen_lag1/ccomisiones_otras","mactivos_margen_lag1/ccomisiones_otras_lag2","mactivos_margen_lag1/cextraccion_autoservicio","mactivos_margen_lag1/cpayroll_trx","mactivos_margen_lag1/cpayroll_trx_lag1","mactivos_margen_lag1/cpayroll_trx_lag2","mactivos_margen_lag1/cprestamos_personales","mactivos_margen_lag1/cproductos","mactivos_margen_lag1/cproductos_delta1","mactivos_margen_lag1/cproductos_delta2","mactivos_margen_lag1/ctarjeta_debito_transacciones","mactivos_margen_lag1/ctarjeta_debito_transacciones_lag1","mactivos_margen_lag1/ctarjeta_visa_transacciones","mactivos_margen_lag1/ctarjeta_visa_transacciones_lag1","mactivos_margen_lag1/ctarjeta_visa_transacciones_lag2","mactivos_margen_lag2","mactivos_margen_lag2/ccaja_ahorro")

campos_malos3bisbis <- c("mactivos_margen_lag2/ccomisiones_otras","mactivos_margen_lag2/ccomisiones_otras_lag2","mactivos_margen_lag2/cextraccion_autoservicio","ctarjeta_visa_transacciones/ccomisiones_otras","ctarjeta_visa_transacciones/ccomisiones_otras_lag2","ctarjeta_visa_transacciones/cextraccion_autoservicio","ctarjeta_visa_transacciones/cpayroll_trx_lag2","ctarjeta_visa_transacciones/cprestamos_personales","ctarjeta_visa_transacciones/cproductos","ctarjeta_visa_transacciones/cproductos_delta1","ctarjeta_visa_transacciones/cproductos_delta2","cextraccion_autoservicio","cextraccion_autoservicio_tend1","cextraccion_autoservicio/ccaja_ahorro","cextraccion_autoservicio/ccomisiones_otras","cextraccion_autoservicio/ccomisiones_otras_lag2","cextraccion_autoservicio/cpayroll_trx","cextraccion_autoservicio/cpayroll_trx_lag1","cextraccion_autoservicio/cpayroll_trx_lag2","cextraccion_autoservicio/cprestamos_personales","cextraccion_autoservicio/cproductos","cextraccion_autoservicio/cproductos_delta1","cextraccion_autoservicio/cproductos_delta2","cextraccion_autoservicio/ctarjeta_debito_transacciones","cextraccion_autoservicio/ctarjeta_debito_transacciones_lag1","cextraccion_autoservicio/ctarjeta_visa_transacciones_lag1","cextraccion_autoservicio/ctarjeta_visa_transacciones_lag2","chomebanking_transacciones_lag1","chomebanking_transacciones_lag2","cliente_antiguedad","cliente_antiguedad_lag1","cliente_antiguedad_lag2","cpagodeservicios","cpayroll_trx","cpayroll_trx_lag1","cpayroll_trx_lag1/ccaja_ahorro","cpayroll_trx_lag1/ccomisiones_otras","cpayroll_trx_lag1/ccomisiones_otras_lag2","cpayroll_trx_lag1/cextraccion_autoservicio","cpayroll_trx_lag1/cpayroll_trx","cpayroll_trx_lag1/cpayroll_trx_lag2","cpayroll_trx_lag1/cprestamos_personales","cpayroll_trx_lag1/cproductos","cpayroll_trx_lag1/cproductos_delta1","cpayroll_trx_lag1/cproductos_delta2","cpayroll_trx_lag1/ctarjeta_debito_transacciones_lag1","cpayroll_trx_lag1/ctarjeta_visa_transacciones","cpayroll_trx_lag1/ctarjeta_visa_transacciones_lag1","cpayroll_trx_lag1/ctarjeta_visa_transacciones_lag2","cpayroll_trx_lag2","cpayroll_trx_lag2/ccaja_ahorro","cpayroll_trx_lag2/ccomisiones_otras","cpayroll_trx_lag2/ccomisiones_otras_lag2","cpayroll_trx_lag2/cextraccion_autoservicio","cpayroll_trx_lag2/cpayroll_trx","cpayroll_trx_lag2/cpayroll_trx_lag1","cpayroll_trx_lag2/cprestamos_personales","cpayroll_trx_lag2/cproductos","cpayroll_trx_lag2/cproductos_delta1","cpayroll_trx_lag2/cproductos_delta2","cpayroll_trx_lag2/ctarjeta_debito_transacciones","cpayroll_trx_lag2/ctarjeta_debito_transacciones_lag1","cpayroll_trx_lag2/ctarjeta_visa_transacciones","cpayroll_trx_lag2/ctarjeta_visa_transacciones_lag1","cpayroll_trx_lag2/ctarjeta_visa_transacciones_lag2","cpayroll_trx_tend1","cpayroll_trx/ccomisiones_otras","cpayroll_trx/ccomisiones_otras_lag2","cpayroll_trx/cextraccion_autoservicio","cpayroll_trx/cpayroll_trx_lag1","cpayroll_trx/cpayroll_trx_lag2","cpayroll_trx/cproductos","cpayroll_trx/cproductos_delta1","cpayroll_trx/cproductos_delta2","cpayroll_trx/ctarjeta_debito_transacciones_lag1","cpayroll_trx/ctarjeta_visa_transacciones_lag1","cpayroll_trx/ctarjeta_visa_transacciones_lag2","cprestamos_personales","cprestamos_personales_tend1","cprestamos_personales/ccaja_ahorro","cprestamos_personales/ccomisiones_otras","cprestamos_personales/ccomisiones_otras_lag2","cprestamos_personales/cextraccion_autoservicio","cprestamos_personales/cpayroll_trx","cprestamos_personales/cpayroll_trx_lag1","cprestamos_personales/cpayroll_trx_lag2","cprestamos_personales/cproductos","cprestamos_personales/cproductos_delta1","cprestamos_personales/cproductos_delta2","cprestamos_personales/ctarjeta_debito_transacciones")

campos_malos4 <- c("mprestamos_personales_lag2/cpayroll_trx","mprestamos_personales_lag2/cpayroll_trx_lag1","mprestamos_personales_lag2/cpayroll_trx_lag2","mprestamos_personales_lag2/cprestamos_personales","mprestamos_personales_lag2/cproductos","mprestamos_personales_lag2/cproductos_delta1","mprestamos_personales_lag2/cproductos_delta2","mprestamos_personales_lag2/ctarjeta_debito_transacciones","mprestamos_personales_lag2/ctarjeta_debito_transacciones_lag1","mprestamos_personales_lag2/ctarjeta_visa_transacciones","mprestamos_personales_lag2/ctarjeta_visa_transacciones_lag1","mprestamos_personales_lag2/ctarjeta_visa_transacciones_lag2","mprestamos_personales/ccomisiones_otras","mprestamos_personales/ccomisiones_otras_lag2","mprestamos_personales/cextraccion_autoservicio","mprestamos_personales/cpayroll_trx_lag1","mprestamos_personales/cpayroll_trx_lag2","mprestamos_personales/cprestamos_personales","mprestamos_personales/cproductos_delta1","mprestamos_personales/cproductos_delta2","mprestamos_personales/ctarjeta_debito_transacciones","mprestamos_personales/ctarjeta_debito_transacciones_lag1","mprestamos_personales/ctarjeta_visa_transacciones","mprestamos_personales/ctarjeta_visa_transacciones_lag2","mrentabilidad","mrentabilidad_annual_lag1","mrentabilidad_annual_lag2/ccomisiones_otras","mrentabilidad_annual_lag2/ccomisiones_otras_lag2","mrentabilidad_annual_lag2/cextraccion_autoservicio","mrentabilidad_annual_lag2/cpayroll_trx","mrentabilidad_annual_lag2/cpayroll_trx_lag1","mrentabilidad_annual_lag2/cpayroll_trx_lag2","mrentabilidad_annual_lag2/cprestamos_personales","mrentabilidad_annual_lag2/cproductos","mrentabilidad_annual_lag2/cproductos_delta1","mrentabilidad_annual_lag2/cproductos_delta2","mrentabilidad_annual_lag2/ctarjeta_debito_transacciones","mrentabilidad_annual_lag2/ctarjeta_debito_transacciones_lag1","mrentabilidad_annual_lag2/ctarjeta_visa_transacciones","mrentabilidad_annual_lag2/ctarjeta_visa_transacciones_lag1","mrentabilidad_annual_lag2/ctarjeta_visa_transacciones_lag2","mrentabilidad_annual/ccomisiones_otras","mrentabilidad_annual/ccomisiones_otras_lag2","mrentabilidad_annual/cextraccion_autoservicio","mrentabilidad_annual/cpayroll_trx","mrentabilidad_annual/cpayroll_trx_lag1","mrentabilidad_annual/cpayroll_trx_lag2","mrentabilidad_annual/cprestamos_personales","mrentabilidad_annual/cproductos","mrentabilidad_annual/cproductos_delta1","mrentabilidad_annual/cproductos_delta2","mrentabilidad_annual/ctarjeta_debito_transacciones","mrentabilidad_annual/ctarjeta_debito_transacciones_lag1","mrentabilidad_annual/ctarjeta_visa_transacciones","mrentabilidad_annual/ctarjeta_visa_transacciones_lag1","mrentabilidad_annual/ctarjeta_visa_transacciones_lag2","mrentabilidad_delta1","mrentabilidad_delta2","mrentabilidad_lag1","mrentabilidad_lag1/ccomisiones_otras","mrentabilidad_lag1/ccomisiones_otras_lag2","mrentabilidad_lag1/cextraccion_autoservicio","mrentabilidad_lag1/cpayroll_trx","mrentabilidad_lag1/cpayroll_trx_lag1","mrentabilidad_lag1/cpayroll_trx_lag2","mrentabilidad_lag1/cprestamos_personales","mrentabilidad_lag1/cproductos","mrentabilidad_lag1/cproductos_delta1","mrentabilidad_lag1/cproductos_delta2","mrentabilidad_lag1/ctarjeta_debito_transacciones","mrentabilidad_lag1/ctarjeta_debito_transacciones_lag1","mrentabilidad_lag1/ctarjeta_visa_transacciones","mrentabilidad_lag1/ctarjeta_visa_transacciones_lag1","mrentabilidad_lag1/ctarjeta_visa_transacciones_lag2","mrentabilidad_lag2","mrentabilidad_lag2/ccomisiones_otras","mrentabilidad_lag2/ccomisiones_otras_lag2","mrentabilidad_lag2/cextraccion_autoservicio","mrentabilidad_lag2/cpayroll_trx","mrentabilidad_lag2/cpayroll_trx_lag1","mrentabilidad_lag2/cpayroll_trx_lag2","mrentabilidad_lag2/cprestamos_personales","mrentabilidad_lag2/cproductos","mrentabilidad_lag2/cproductos_delta1","mrentabilidad_lag2/cproductos_delta2")

campos_malos5 <- c("mrentabilidad_lag2/ctarjeta_debito_transacciones","mrentabilidad_lag2/ctarjeta_debito_transacciones_lag1","mrentabilidad_lag2/ctarjeta_visa_transacciones","mrentabilidad_lag2/ctarjeta_visa_transacciones_lag1","mrentabilidad_lag2/ctarjeta_visa_transacciones_lag2","mrentabilidad_tend1","mrentabilidad_tend1/ccomisiones_otras","mrentabilidad_tend1/cextraccion_autoservicio","mrentabilidad_tend1/cpayroll_trx","mrentabilidad_tend1/cpayroll_trx_lag1","mrentabilidad_tend1/cpayroll_trx_lag2","mrentabilidad_tend1/cprestamos_personales","mrentabilidad_tend1/cproductos_delta1","mrentabilidad_tend1/cproductos_delta2","mrentabilidad_tend1/ctarjeta_debito_transacciones","mrentabilidad_tend1/ctarjeta_debito_transacciones_lag1","mrentabilidad_tend1/ctarjeta_visa_transacciones","mrentabilidad_tend1/ctarjeta_visa_transacciones_lag1","mrentabilidad_tend1/ctarjeta_visa_transacciones_lag2","mrentabilidad/ccomisiones_otras","mrentabilidad/ccomisiones_otras_lag2","mrentabilidad/cextraccion_autoservicio","mrentabilidad/cpayroll_trx","mrentabilidad/cpayroll_trx_lag1","mrentabilidad/cpayroll_trx_lag2","mrentabilidad/cprestamos_personales","mrentabilidad/cproductos_delta1","mrentabilidad/cproductos_delta2","mrentabilidad/ctarjeta_debito_transacciones","mrentabilidad/ctarjeta_debito_transacciones_lag1","mrentabilidad/ctarjeta_visa_transacciones","mrentabilidad/ctarjeta_visa_transacciones_lag1","mrentabilidad/ctarjeta_visa_transacciones_lag2","mtarjeta_visa_consumo_lag1","mtarjeta_visa_consumo_lag1/ccaja_ahorro","mtarjeta_visa_consumo_lag1/ccomisiones_otras","mtarjeta_visa_consumo_lag1/ccomisiones_otras_lag2","mtarjeta_visa_consumo_lag1/cextraccion_autoservicio","mtarjeta_visa_consumo_lag1/cpayroll_trx","mtarjeta_visa_consumo_lag1/cpayroll_trx_lag1","mtarjeta_visa_consumo_lag1/cpayroll_trx_lag2","mtarjeta_visa_consumo_lag1/cprestamos_personales","mtarjeta_visa_consumo_lag1/cproductos","mtarjeta_visa_consumo_lag1/cproductos_delta1","mtarjeta_visa_consumo_lag1/cproductos_delta2","mtarjeta_visa_consumo_lag1/ctarjeta_debito_transacciones","mtarjeta_visa_consumo_lag1/ctarjeta_debito_transacciones_lag1","mtarjeta_visa_consumo_lag1/ctarjeta_visa_transacciones","mtarjeta_visa_consumo_lag1/ctarjeta_visa_transacciones_lag1","mtarjeta_visa_consumo_lag1/ctarjeta_visa_transacciones_lag2","mtarjeta_visa_consumo/ccomisiones_otras","mtarjeta_visa_consumo/ccomisiones_otras_lag2","mtarjeta_visa_consumo/cextraccion_autoservicio","mtarjeta_visa_consumo/cpayroll_trx_lag1","mtarjeta_visa_consumo/cpayroll_trx_lag2","mtarjeta_visa_consumo/cprestamos_personales","mtarjeta_visa_consumo/cproductos","mtarjeta_visa_consumo/cproductos_delta1","mtarjeta_visa_consumo/cproductos_delta2","mtarjeta_visa_consumo/ctarjeta_debito_transacciones_lag1","mtarjeta_visa_consumo/ctarjeta_visa_transacciones","mtarjeta_visa_consumo/ctarjeta_visa_transacciones_lag1","mtarjeta_visa_consumo/ctarjeta_visa_transacciones_lag2","mtarjeta_visa_consumo/mactivos_margen","mtarjeta_visa_consumo/mactivos_margen_lag1","mtarjeta_visa_consumo/mactivos_margen_lag2","mtarjeta_visa_consumo/Master_fechaalta_lag2","mtarjeta_visa_consumo/Master_mlimitecompra_lag2","mtarjeta_visa_consumo/mcaja_ahorro_dolares","mtarjeta_visa_consumo/mcomisiones_lag2","mtarjeta_visa_consumo/mcomisiones_otras","mtarjeta_visa_consumo/mcuenta_corriente","mtarjeta_visa_consumo/mcuenta_corriente_lag1","mtarjeta_visa_consumo/mpayroll_tend1","mtarjeta_visa_consumo/mprestamos_personales","mtarjeta_visa_consumo/mprestamos_personales_lag2","mtarjeta_visa_consumo/mrentabilidad","mtarjeta_visa_consumo/mrentabilidad_annual")

campos_malos6 <- c("mtarjeta_visa_consumo/mv_msaldopesos","mtarjeta_visa_consumo/mv_msaldototal","mtarjeta_visa_consumo/mv_status04","mtarjeta_visa_consumo/mvr_mpagospesos","mtarjeta_visa_consumo/mvr_msaldopesos","mtarjeta_visa_consumo/mvr_msaldototal","mtarjeta_visa_consumo/Visa_mlimitecompra","mtarjeta_visa_consumo/Visa_mpagominimo","mtarjeta_visa_consumo/Visa_msaldototal","mtarjeta_visa_consumo/Visa_msaldototal_delta1","mtarjeta_visa_consumo/Visa_msaldototal_lag1","mtransferencias_recibidas_lag1","mtransferencias_recibidas_lag1/ccaja_ahorro","mtransferencias_recibidas_lag1/ccomisiones_otras","mtransferencias_recibidas_lag1/ccomisiones_otras_lag2","mtransferencias_recibidas_lag1/cextraccion_autoservicio","mtransferencias_recibidas_lag1/cpayroll_trx","mtransferencias_recibidas_lag1/cpayroll_trx_lag1","mtransferencias_recibidas_lag1/cpayroll_trx_lag2","mtransferencias_recibidas_lag1/cprestamos_personales","mtransferencias_recibidas_lag1/cproductos","mtransferencias_recibidas_lag1/cproductos_delta1","mtransferencias_recibidas_lag1/cproductos_delta2","mtransferencias_recibidas_lag1/ctarjeta_debito_transacciones","mtransferencias_recibidas_lag1/ctarjeta_debito_transacciones_lag1","mtransferencias_recibidas_lag1/ctarjeta_visa_transacciones","mtransferencias_recibidas_lag1/ctarjeta_visa_transacciones_lag1","mtransferencias_recibidas_lag1/ctarjeta_visa_transacciones_lag2","mv_cadelantosefectivo","mv_fechaalta","mv_fultimo_cierre","mv_madelantopesos","mv_mconsumospesos","mv_mconsumospesos/ccaja_ahorro","mv_mconsumospesos/ccomisiones_otras","mv_mconsumospesos/ccomisiones_otras_lag2","mv_mconsumospesos/cextraccion_autoservicio","mv_mconsumospesos/cpayroll_trx","mv_mconsumospesos/cpayroll_trx_lag1","mv_mconsumospesos/cpayroll_trx_lag2","mv_mconsumospesos/cprestamos_personales","mv_mconsumospesos/cproductos","mv_mconsumospesos/cproductos_delta1","mv_mconsumospesos/cproductos_delta2","mv_mconsumospesos/ctarjeta_debito_transacciones","mv_mconsumospesos/ctarjeta_debito_transacciones_lag1","mv_mconsumospesos/ctarjeta_visa_transacciones","mv_mconsumospesos/ctarjeta_visa_transacciones_lag1","mv_mconsumospesos/ctarjeta_visa_transacciones_lag2","mv_mconsumototal","mv_mconsumototal/ccaja_ahorro","mv_mconsumototal/ccomisiones_otras","mv_mconsumototal/ccomisiones_otras_lag2","mv_mconsumototal/cextraccion_autoservicio","mv_mconsumototal/cpayroll_trx_lag2","mv_mconsumototal/cprestamos_personales","mv_mconsumototal/cproductos")

campos_malos7 <- c("mv_mconsumototal/cproductos_delta1","mv_mconsumototal/cproductos_delta2","mv_mconsumototal/ctarjeta_debito_transacciones","mv_mconsumototal/ctarjeta_debito_transacciones_lag1","mv_mconsumototal/ctarjeta_visa_transacciones","mv_mconsumototal/ctarjeta_visa_transacciones_lag1","mv_mconsumototal/ctarjeta_visa_transacciones_lag2","mv_mfinanciacion_limite","mv_mlimitecompra","mv_mpagominimo","mv_mpagominimo/ccaja_ahorro","mv_mpagominimo/ccomisiones_otras","mv_mpagominimo/ccomisiones_otras_lag2","mv_mpagominimo/cextraccion_autoservicio","mv_mpagominimo/cpayroll_trx_lag1","mv_mpagominimo/cpayroll_trx_lag2","mv_mpagominimo/cprestamos_personales","mv_mpagominimo/cproductos","mv_mpagominimo/cproductos_delta1","mv_mpagominimo/cproductos_delta2","mv_mpagominimo/ctarjeta_debito_transacciones","mv_mpagominimo/ctarjeta_debito_transacciones_lag1","mv_mpagominimo/ctarjeta_visa_transacciones","mv_mpagominimo/ctarjeta_visa_transacciones_lag1","mv_mpagominimo/ctarjeta_visa_transacciones_lag2","mv_mpagospesos","mv_mpagospesos/ccaja_ahorro","mv_mpagospesos/ccomisiones_otras","mv_mpagospesos/ccomisiones_otras_lag2","mv_mpagospesos/cextraccion_autoservicio","mv_mpagospesos/cpayroll_trx","mv_mpagospesos/cpayroll_trx_lag1","mv_mpagospesos/cpayroll_trx_lag2","mv_mpagospesos/cprestamos_personales","mv_mpagospesos/cproductos","mv_mpagospesos/cproductos_delta1","mv_mpagospesos/cproductos_delta2","mv_mpagospesos/ctarjeta_debito_transacciones","mv_mpagospesos/ctarjeta_debito_transacciones_lag1","mv_mpagospesos/ctarjeta_visa_transacciones","mv_mpagospesos/ctarjeta_visa_transacciones_lag1","mv_mpagospesos/ctarjeta_visa_transacciones_lag2","mv_msaldopesos","mv_msaldopesos/ccaja_ahorro","mv_msaldopesos/ccomisiones_otras","mv_msaldopesos/ccomisiones_otras_lag2","mv_msaldopesos/cextraccion_autoservicio","mv_msaldopesos/cpayroll_trx","mv_msaldopesos/cpayroll_trx_lag1","mv_msaldopesos/cpayroll_trx_lag2","mv_msaldopesos/cprestamos_personales","mv_msaldopesos/cproductos","mv_msaldopesos/cproductos_delta1","mv_msaldopesos/cproductos_delta2")

campos_malos8 <- c("mv_mconsumospesos/ctarjeta_visa_transacciones_lag1","mv_mconsumospesos/ctarjeta_visa_transacciones","mv_mconsumospesos/ctarjeta_debito_transacciones_lag1","mv_mconsumospesos/ctarjeta_debito_transacciones","mv_mconsumospesos/cproductos_delta2","mv_mconsumospesos/cproductos_delta1","mv_mconsumospesos/cproductos","mv_mconsumospesos/cprestamos_personales","mv_mconsumospesos/cpayroll_trx_lag2","mv_mconsumospesos/cpayroll_trx_lag1","mv_mconsumospesos/cpayroll_trx","mv_mconsumospesos/cextraccion_autoservicio","mv_mconsumospesos/ccomisiones_otras_lag2","mv_mconsumospesos/ccomisiones_otras","mv_mconsumospesos/ccaja_ahorro","mv_mconsumospesos","mv_madelantopesos","mv_fultimo_cierre","mv_fechaalta","mv_cadelantosefectivo","mtransferencias_recibidas_lag1/ctarjeta_visa_transacciones_lag2","mtransferencias_recibidas_lag1/ctarjeta_visa_transacciones_lag1","mtransferencias_recibidas_lag1/ctarjeta_visa_transacciones","mtransferencias_recibidas_lag1/ctarjeta_debito_transacciones_lag1","mtransferencias_recibidas_lag1/ctarjeta_debito_transacciones","mtransferencias_recibidas_lag1/cproductos_delta2","mtransferencias_recibidas_lag1/cproductos_delta1","mtransferencias_recibidas_lag1/cproductos","mtransferencias_recibidas_lag1/cprestamos_personales","mtransferencias_recibidas_lag1/cpayroll_trx_lag2","mtransferencias_recibidas_lag1/cpayroll_trx_lag1","mtransferencias_recibidas_lag1/cpayroll_trx","mtransferencias_recibidas_lag1/cextraccion_autoservicio","mtransferencias_recibidas_lag1/ccomisiones_otras_lag2","mtransferencias_recibidas_lag1/ccomisiones_otras","mtransferencias_recibidas_lag1/ccaja_ahorro","mtransferencias_recibidas_lag1","mtarjeta_visa_consumo/Visa_msaldototal_lag1","mtarjeta_visa_consumo/Visa_msaldototal_delta1","mtarjeta_visa_consumo/Visa_msaldototal","mtarjeta_visa_consumo/Visa_mpagominimo","mtarjeta_visa_consumo/Visa_mlimitecompra","mtarjeta_visa_consumo/mvr_msaldototal","mtarjeta_visa_consumo/mvr_msaldopesos","mtarjeta_visa_consumo/mvr_mpagospesos","mtarjeta_visa_consumo/mv_status04","mtarjeta_visa_consumo/mv_msaldototal","mtarjeta_visa_consumo/mv_msaldopesos","mtarjeta_visa_consumo/mv_mpagospesos","mtarjeta_visa_consumo/mv_mpagominimo","mtarjeta_visa_consumo/mv_mconsumototal","mtarjeta_visa_consumo/mv_mconsumospesos","mtarjeta_visa_consumo/mtransferencias_recibidas_lag1","mtarjeta_visa_consumo/mtarjeta_visa_consumo_lag1","mtarjeta_visa_consumo/mrentabilidad_tend1","mtarjeta_visa_consumo/mrentabilidad_lag2","mtarjeta_visa_consumo/mrentabilidad_lag1","mtarjeta_visa_consumo/mrentabilidad_annual_lag2","mtarjeta_visa_consumo/mrentabilidad_annual","mtarjeta_visa_consumo/mrentabilidad","mtarjeta_visa_consumo/mprestamos_personales_lag2","mtarjeta_visa_consumo/mprestamos_personales","mtarjeta_visa_consumo/mpayroll_tend1","mtarjeta_visa_consumo/mcuenta_corriente_lag1","mtarjeta_visa_consumo/mcuenta_corriente","mtarjeta_visa_consumo/mcomisiones_otras","mtarjeta_visa_consumo/mcomisiones_lag2","mtarjeta_visa_consumo/mcaja_ahorro_dolares","mtarjeta_visa_consumo/Master_mlimitecompra_lag2","mtarjeta_visa_consumo/Master_fechaalta_lag2","mtarjeta_visa_consumo/mactivos_margen_lag2","mtarjeta_visa_consumo/mactivos_margen_lag1","mtarjeta_visa_consumo/mactivos_margen","mtarjeta_visa_consumo/ctarjeta_visa_transacciones_lag2","mtarjeta_visa_consumo/ctarjeta_visa_transacciones_lag1","mtarjeta_visa_consumo/ctarjeta_visa_transacciones","mtarjeta_visa_consumo/ctarjeta_debito_transacciones_lag1","mtarjeta_visa_consumo/cproductos_delta2","mtarjeta_visa_consumo/cproductos_delta1","mtarjeta_visa_consumo/cproductos","mtarjeta_visa_consumo/cprestamos_personales","mtarjeta_visa_consumo/cpayroll_trx_lag2","mtarjeta_visa_consumo/cpayroll_trx_lag1","mtarjeta_visa_consumo/cextraccion_autoservicio")

campos_malos8bis<- c("mrentabilidad_annual_lag2/ctarjeta_debito_transacciones_lag1","mrentabilidad_annual_lag2/ctarjeta_debito_transacciones","mrentabilidad_annual_lag2/cproductos_delta2","mrentabilidad_annual_lag2/cproductos_delta1","mrentabilidad_annual_lag2/cproductos","mrentabilidad_annual_lag2/cprestamos_personales","mrentabilidad_annual_lag2/cpayroll_trx_lag2","mrentabilidad_annual_lag2/cpayroll_trx_lag1","mrentabilidad_annual_lag2/cpayroll_trx","mrentabilidad_annual_lag2/cextraccion_autoservicio","mrentabilidad_annual_lag2/ccomisiones_otras_lag2","mrentabilidad_annual_lag2/ccomisiones_otras","mrentabilidad_annual_lag1","mrentabilidad","mprestamos_personales/ctarjeta_visa_transacciones_lag2","mprestamos_personales/ctarjeta_visa_transacciones","mprestamos_personales/ctarjeta_debito_transacciones_lag1","mprestamos_personales/ctarjeta_debito_transacciones","mprestamos_personales/cproductos_delta2","mprestamos_personales/cproductos_delta1","mprestamos_personales/cprestamos_personales","mprestamos_personales/cpayroll_trx_lag2","mprestamos_personales/cpayroll_trx_lag1","mprestamos_personales/cextraccion_autoservicio","mprestamos_personales/ccomisiones_otras_lag2","mprestamos_personales/ccomisiones_otras","mprestamos_personales_lag2/ctarjeta_visa_transacciones_lag2","mprestamos_personales_lag2/ctarjeta_visa_transacciones_lag1","mprestamos_personales_lag2/ctarjeta_visa_transacciones","mprestamos_personales_lag2/ctarjeta_debito_transacciones_lag1","mprestamos_personales_lag2/ctarjeta_debito_transacciones","mprestamos_personales_lag2/cproductos_delta2","mprestamos_personales_lag2/cproductos_delta1","mprestamos_personales_lag2/cproductos","mprestamos_personales_lag2/cprestamos_personales","mprestamos_personales_lag2/cpayroll_trx_lag2","mprestamos_personales_lag2/cpayroll_trx_lag1","mprestamos_personales_lag2/cpayroll_trx","mprestamos_personales_lag2/cextraccion_autoservicio","mprestamos_personales_lag2/ccomisiones_otras_lag2","mprestamos_personales_lag2/ccomisiones_otras","mprestamos_personales_lag2","mprestamos_personales_lag1","mprestamos_personales_delta1","mprestamos_personales","mtarjeta_visa_consumo/mrentabilidad_annual_lag2","mtarjeta_visa_consumo/mrentabilidad_lag1","mtarjeta_visa_consumo/mrentabilidad_lag2","mtarjeta_visa_consumo/mrentabilidad_tend1","mtarjeta_visa_consumo/mtarjeta_visa_consumo_lag1","mtarjeta_visa_consumo/mtransferencias_recibidas_lag1","mtarjeta_visa_consumo/mv_mconsumospesos","mtarjeta_visa_consumo/mv_mconsumototal","mtarjeta_visa_consumo/mv_mpagominimo","mtarjeta_visa_consumo/mv_mpagospesos")

campos_malos8bisbis <- c("mtarjeta_visa_consumo/ccomisiones_otras_lag2","mtarjeta_visa_consumo/ccomisiones_otras","mtarjeta_visa_consumo_lag1/ctarjeta_visa_transacciones_lag2","mtarjeta_visa_consumo_lag1/ctarjeta_visa_transacciones_lag1","mtarjeta_visa_consumo_lag1/ctarjeta_visa_transacciones","mtarjeta_visa_consumo_lag1/ctarjeta_debito_transacciones_lag1","mtarjeta_visa_consumo_lag1/ctarjeta_debito_transacciones","mtarjeta_visa_consumo_lag1/cproductos_delta2","mtarjeta_visa_consumo_lag1/cproductos_delta1","mtarjeta_visa_consumo_lag1/cproductos","mtarjeta_visa_consumo_lag1/cprestamos_personales","mtarjeta_visa_consumo_lag1/cpayroll_trx_lag2","mtarjeta_visa_consumo_lag1/cpayroll_trx_lag1","mtarjeta_visa_consumo_lag1/cpayroll_trx","mtarjeta_visa_consumo_lag1/cextraccion_autoservicio","mtarjeta_visa_consumo_lag1/ccomisiones_otras_lag2","mtarjeta_visa_consumo_lag1/ccomisiones_otras","mtarjeta_visa_consumo_lag1/ccaja_ahorro","mtarjeta_visa_consumo_lag1","mrentabilidad/ctarjeta_visa_transacciones_lag2","mrentabilidad/ctarjeta_visa_transacciones_lag1","mrentabilidad/ctarjeta_visa_transacciones","mrentabilidad/ctarjeta_debito_transacciones_lag1","mrentabilidad/ctarjeta_debito_transacciones","mrentabilidad/cproductos_delta2","mrentabilidad/cproductos_delta1","mrentabilidad/cprestamos_personales","mrentabilidad/cpayroll_trx_lag2","mrentabilidad/cpayroll_trx_lag1","mrentabilidad/cpayroll_trx","mrentabilidad/cextraccion_autoservicio","mrentabilidad/ccomisiones_otras_lag2","mrentabilidad/ccomisiones_otras","mrentabilidad_tend1/ctarjeta_visa_transacciones_lag2","mrentabilidad_tend1/ctarjeta_visa_transacciones_lag1","mrentabilidad_tend1/ctarjeta_visa_transacciones","mrentabilidad_tend1/ctarjeta_debito_transacciones_lag1","mrentabilidad_tend1/ctarjeta_debito_transacciones","mrentabilidad_tend1/cproductos_delta2","mrentabilidad_tend1/cproductos_delta1","mrentabilidad_tend1/cprestamos_personales","mrentabilidad_tend1/cpayroll_trx_lag2","mrentabilidad_tend1/cpayroll_trx_lag1","mrentabilidad_tend1/cpayroll_trx","mrentabilidad_tend1/cextraccion_autoservicio","mrentabilidad_tend1/ccomisiones_otras","mrentabilidad_tend1","mrentabilidad_lag2/ctarjeta_visa_transacciones_lag2","mrentabilidad_lag2/ctarjeta_visa_transacciones_lag1","mrentabilidad_lag2/ctarjeta_visa_transacciones","mrentabilidad_lag2/ctarjeta_debito_transacciones_lag1","mrentabilidad_lag2/ctarjeta_debito_transacciones","mrentabilidad_lag2/cproductos_delta2","mrentabilidad_lag2/cproductos_delta1","mrentabilidad_lag2/cproductos","mrentabilidad_lag2/cprestamos_personales","mrentabilidad_lag2/cpayroll_trx_lag2","mrentabilidad_lag2/cpayroll_trx_lag1","mrentabilidad_lag2/cpayroll_trx","mrentabilidad_lag2/cextraccion_autoservicio","mrentabilidad_lag2/ccomisiones_otras_lag2","mrentabilidad_lag2/ccomisiones_otras","mrentabilidad_lag2","mrentabilidad_lag1/ctarjeta_visa_transacciones_lag2","mrentabilidad_lag1/ctarjeta_visa_transacciones_lag1","mrentabilidad_lag1/ctarjeta_visa_transacciones","mrentabilidad_lag1/ctarjeta_debito_transacciones_lag1","mrentabilidad_lag1/ctarjeta_debito_transacciones","mrentabilidad_lag1/cproductos_delta2","mrentabilidad_lag1/cproductos_delta1","mrentabilidad_lag1/cproductos","mrentabilidad_lag1/cprestamos_personales","mrentabilidad_lag1/cpayroll_trx_lag2","mrentabilidad_lag1/cpayroll_trx_lag1","mrentabilidad_lag1/cpayroll_trx","mrentabilidad_lag1/cextraccion_autoservicio","mrentabilidad_lag1/ccomisiones_otras_lag2","mrentabilidad_lag1/ccomisiones_otras","mrentabilidad_lag1","mrentabilidad_delta2","mrentabilidad_delta1","mrentabilidad_annual/ctarjeta_visa_transacciones_lag2","mrentabilidad_annual/ctarjeta_visa_transacciones_lag1","mrentabilidad_annual/ctarjeta_visa_transacciones","mrentabilidad_annual/ctarjeta_debito_transacciones_lag1")

campos_malos9 <- c("mplazo_fijo_pesos_delta2","mpayroll2_tend1","mpayroll_tend1/ctarjeta_visa_transacciones_lag2","mpayroll_tend1/ctarjeta_visa_transacciones_lag1","mpayroll_tend1/ctarjeta_visa_transacciones","mpayroll_tend1/ctarjeta_debito_transacciones_lag1","mpayroll_tend1/ctarjeta_debito_transacciones","mpayroll_tend1/cproductos_delta2","mpayroll_tend1/cproductos_delta1","mpayroll_tend1/cproductos","mpayroll_tend1/cprestamos_personales","mpayroll_tend1/cpayroll_trx_lag2","mpayroll_tend1/cpayroll_trx_lag1","mpayroll_tend1/cpayroll_trx","mpayroll_tend1/cextraccion_autoservicio","mpayroll_tend1/ccomisiones_otras_lag2","mpayroll_tend1/ccomisiones_otras","mpayroll_tend1/ccaja_ahorro","mpayroll_tend1","mpayroll_lag2","mpayroll_lag1","mpayroll_delta2","mpayroll_delta1","mpayroll","mpagodeservicios_lag1","mpagodeservicios_delta2","mpagodeservicios_delta1","mpagodeservicios","minversion1_dolares_lag2","minversion1_dolares_lag1","mdescubierto_preacordado/ctarjeta_visa_transacciones_lag2","mdescubierto_preacordado/ctarjeta_visa_transacciones_lag1","mdescubierto_preacordado/ctarjeta_visa_transacciones","mdescubierto_preacordado/ctarjeta_debito_transacciones_lag1","mdescubierto_preacordado/ctarjeta_debito_transacciones","mdescubierto_preacordado/cproductos_delta2","mdescubierto_preacordado/cproductos_delta1","mdescubierto_preacordado/cprestamos_personales","mdescubierto_preacordado/cpayroll_trx_lag2","mdescubierto_preacordado/cpayroll_trx_lag1","mdescubierto_preacordado/cextraccion_autoservicio","mdescubierto_preacordado/ccomisiones_otras_lag2","mdescubierto_preacordado/ccomisiones_otras","mdescubierto_preacordado_lag1","mdescubierto_preacordado","mcuenta_corriente/ctarjeta_visa_transacciones_lag2","mcuenta_corriente/ctarjeta_visa_transacciones_lag1","mcuenta_corriente/ctarjeta_visa_transacciones","mcuenta_corriente/ctarjeta_debito_transacciones","mcuenta_corriente/cproductos_delta2","mcuenta_corriente/cproductos_delta1","mcuenta_corriente/cproductos","mcuenta_corriente/cpayroll_trx_lag2","mcuenta_corriente/cpayroll_trx_lag1","mcuenta_corriente/ccaja_ahorro","mcuenta_corriente_lag2","mcuenta_corriente_lag1/ctarjeta_visa_transacciones_lag2","mcuenta_corriente_lag1/ctarjeta_visa_transacciones_lag1","mcuenta_corriente_lag1/ctarjeta_visa_transacciones","mcuenta_corriente_lag1/ctarjeta_debito_transacciones_lag1","mcuenta_corriente_lag1/ctarjeta_debito_transacciones","mcuenta_corriente_lag1/cproductos_delta2","mcuenta_corriente_lag1/cproductos_delta1","mcuenta_corriente_lag1/cproductos","mcuenta_corriente_lag1/cprestamos_personales","mcuenta_corriente_lag1/cpayroll_trx_lag2","mcuenta_corriente_lag1/cpayroll_trx_lag1","mcuenta_corriente_lag1/cpayroll_trx","mcuenta_corriente_lag1/cextraccion_autoservicio","mcuenta_corriente_lag1/ccomisiones_otras_lag2","mcuenta_corriente_lag1/ccomisiones_otras","mcuenta_corriente_lag1/ccaja_ahorro","mcuenta_corriente_lag1","mcuenta_corriente_delta1","mcuenta_corriente","mcomisiones_otras/ctarjeta_visa_transacciones_lag2","mcomisiones_otras/ctarjeta_visa_transacciones_lag1","mcomisiones_otras/ctarjeta_visa_transacciones","mcomisiones_otras/ctarjeta_debito_transacciones_lag1","mcomisiones_otras/ctarjeta_debito_transacciones","mcomisiones_otras/cproductos_delta2","mcomisiones_otras/cproductos_delta1","mcomisiones_otras/cprestamos_personales","mcomisiones_otras/cpayroll_trx_lag2","mcomisiones_otras/cpayroll_trx_lag1","mcomisiones_otras/cpayroll_trx","mcomisiones_otras/cextraccion_autoservicio","mcomisiones_otras/ccomisiones_otras_lag2","mcomisiones_otras/ccomisiones_otras","mcomisiones_otras/ccaja_ahorro","mcomisiones_otras_lag2","mcomisiones_otras_lag1","mcomisiones_otras_delta2","mcomisiones_otras","mcomisiones_mantenimiento_lag2","mcomisiones_lag2/ctarjeta_visa_transacciones_lag2")

campos_malos9bis <- c("Master_mlimitecompra_lag2/cproductos","Master_mlimitecompra_lag2/cprestamos_personales","Master_mlimitecompra_lag2/cpayroll_trx_lag2","Master_mlimitecompra_lag2/cpayroll_trx_lag1","Master_mlimitecompra_lag2/cpayroll_trx","Master_mlimitecompra_lag2/cextraccion_autoservicio","Master_mlimitecompra_lag2/ccomisiones_otras_lag2","Master_mlimitecompra_lag2/ccomisiones_otras","Master_mlimitecompra_lag2/ccaja_ahorro","Master_mlimitecompra_lag2","Master_mlimitecompra_lag1","Master_mlimitecompra","Master_Fvencimiento_lag1","Master_fechaalta_tend1","Master_fechaalta_lag2/ctarjeta_visa_transacciones_lag1","Master_fechaalta_lag2/ctarjeta_visa_transacciones","Master_fechaalta_lag2/ctarjeta_debito_transacciones_lag1","Master_fechaalta_lag2/ctarjeta_debito_transacciones","Master_fechaalta_lag2/cproductos_delta2","Master_fechaalta_lag2/cproductos_delta1","Master_fechaalta_lag2/cprestamos_personales","Master_fechaalta_lag2/cpayroll_trx_lag2","Master_fechaalta_lag2/cpayroll_trx_lag1","Master_fechaalta_lag2/cpayroll_trx","Master_fechaalta_lag2/cextraccion_autoservicio","Master_fechaalta_lag2/ccomisiones_otras_lag2","Master_fechaalta_lag2/ccaja_ahorro","Master_fechaalta_lag2","Master_fechaalta_lag1","Master_fechaalta","mactivos_margen/ctarjeta_visa_transacciones","mactivos_margen/ctarjeta_debito_transacciones_lag1","mactivos_margen/ctarjeta_debito_transacciones","mactivos_margen/cproductos_delta2","mactivos_margen/cproductos_delta1","mactivos_margen/cprestamos_personales","mactivos_margen/cpayroll_trx_lag2","mactivos_margen/cpayroll_trx_lag1","mactivos_margen/cpayroll_trx","mactivos_margen/cextraccion_autoservicio","mactivos_margen/ccomisiones_otras_lag2","mactivos_margen/ccomisiones_otras","mactivos_margen_lag2/ctarjeta_visa_transacciones_lag2","mactivos_margen_lag2/ctarjeta_visa_transacciones_lag1","mactivos_margen_lag2/ctarjeta_visa_transacciones","mactivos_margen_lag2/ctarjeta_debito_transacciones_lag1","mactivos_margen_lag2/ctarjeta_debito_transacciones","mactivos_margen_lag2/cproductos_delta2","mactivos_margen_lag2/cproductos_delta1","mactivos_margen_lag2/cproductos","mactivos_margen_lag2/cprestamos_personales","mactivos_margen_lag2/cpayroll_trx_lag2","mactivos_margen_lag2/cpayroll_trx_lag1","mactivos_margen_lag2/cpayroll_trx","mactivos_margen_lag2/cextraccion_autoservicio","mactivos_margen_lag2/ccomisiones_otras_lag2","mactivos_margen_lag2/ccomisiones_otras","mactivos_margen_lag2/ccaja_ahorro","mv_mconsumototal/ctarjeta_visa_transacciones","mv_mconsumototal/ctarjeta_debito_transacciones_lag1","mv_mconsumototal/ctarjeta_debito_transacciones","mv_mconsumototal/cproductos_delta2","mv_mconsumototal/cproductos_delta1","mv_mconsumototal/cproductos","mv_mconsumototal/cprestamos_personales","mv_mconsumototal/cpayroll_trx_lag2","mv_mconsumototal/cextraccion_autoservicio","mv_mconsumototal/ccomisiones_otras_lag2","mv_mconsumototal/ccomisiones_otras","mv_mconsumototal/ccaja_ahorro","mv_mconsumototal","mv_mconsumospesos/ctarjeta_visa_transacciones_lag2")

campos_malos9bisbis <- c("mcomisiones_lag2/ctarjeta_visa_transacciones_lag1","mcomisiones_lag2/ctarjeta_visa_transacciones","mcomisiones_lag2/ctarjeta_debito_transacciones_lag1","mcomisiones_lag2/ctarjeta_debito_transacciones","mcomisiones_lag2/cproductos_delta2","mcomisiones_lag2/cproductos_delta1","mcomisiones_lag2/cproductos","mcomisiones_lag2/cprestamos_personales","mcomisiones_lag2/cpayroll_trx_lag2","mcomisiones_lag2/cpayroll_trx_lag1","mcomisiones_lag2/cpayroll_trx","mcomisiones_lag2/cextraccion_autoservicio","mcomisiones_lag2/ccomisiones_otras_lag2","mcomisiones_lag2/ccaja_ahorro","mcomisiones_lag2","mcomisiones_lag1","mcomisiones_delta2","mcomisiones","mcheques_emitidos_rechazados_lag2","mcheques_emitidos_rechazados_delta1","mcheques_depositados_rechazados","mcaja_ahorro_dolares/ctarjeta_visa_transacciones_lag2","mcaja_ahorro_dolares/ctarjeta_visa_transacciones_lag1","mcaja_ahorro_dolares/ctarjeta_visa_transacciones","mcaja_ahorro_dolares/ctarjeta_debito_transacciones_lag1","mcaja_ahorro_dolares/ctarjeta_debito_transacciones","mcaja_ahorro_dolares/cproductos_delta2","mcaja_ahorro_dolares/cproductos_delta1","mcaja_ahorro_dolares/cproductos","mcaja_ahorro_dolares/cprestamos_personales","mcaja_ahorro_dolares/cpayroll_trx_lag2","mcaja_ahorro_dolares/cpayroll_trx_lag1","mcaja_ahorro_dolares/cpayroll_trx","mcaja_ahorro_dolares/cextraccion_autoservicio","mcaja_ahorro_dolares/ccomisiones_otras_lag2","mcaja_ahorro_dolares/ccomisiones_otras","mcaja_ahorro_dolares/ccaja_ahorro","mcaja_ahorro_dolares_lag2","mcaja_ahorro_dolares_lag1","mcaja_ahorro_dolares","Master_mlimitecompra_lag2/ctarjeta_visa_transacciones_lag2","Master_mlimitecompra_lag2/ctarjeta_visa_transacciones_lag1","Master_mlimitecompra_lag2/ctarjeta_visa_transacciones","Master_mlimitecompra_lag2/ctarjeta_debito_transacciones_lag1","Master_mlimitecompra_lag2/ctarjeta_debito_transacciones","Master_mlimitecompra_lag2/cproductos_delta2","Master_mlimitecompra_lag2/cproductos_delta1","mrentabilidad_annual/ctarjeta_debito_transacciones","mrentabilidad_annual/cproductos_delta2","mrentabilidad_annual/cproductos_delta1","mrentabilidad_annual/cproductos","mrentabilidad_annual/cprestamos_personales","mrentabilidad_annual/cpayroll_trx_lag2","mrentabilidad_annual/cpayroll_trx_lag1","mrentabilidad_annual/cpayroll_trx","mrentabilidad_annual/cextraccion_autoservicio","mrentabilidad_annual/ccomisiones_otras_lag2","mrentabilidad_annual/ccomisiones_otras","mrentabilidad_annual_lag2/ctarjeta_visa_transacciones_lag2","mrentabilidad_annual_lag2/ctarjeta_visa_transacciones_lag1","mrentabilidad_annual_lag2/ctarjeta_visa_transacciones","cprestamos_personales/ctarjeta_debito_transacciones_lag1","cprestamos_personales/ctarjeta_visa_transacciones","cprestamos_personales/ctarjeta_visa_transacciones_lag1","cprestamos_personales/ctarjeta_visa_transacciones_lag2","cproductos_delta1","cproductos_delta1/ccaja_ahorro","cproductos_delta1/ccomisiones_otras","cproductos_delta1/ccomisiones_otras_lag2","cproductos_delta1/cextraccion_autoservicio","cproductos_delta1/cpayroll_trx","cproductos_delta1/cpayroll_trx_lag1","cproductos_delta1/cpayroll_trx_lag2","cproductos_delta1/cprestamos_personales","cproductos_delta1/cproductos","cproductos_delta1/cproductos_delta2")
#estos son nuevos campos malos

campos_malos10 <- c("mactivos_margen_lag2","mactivos_margen_lag1/ctarjeta_visa_transacciones_lag2","mactivos_margen_lag1/ctarjeta_visa_transacciones_lag1","mactivos_margen_lag1/ctarjeta_visa_transacciones","mactivos_margen_lag1/ctarjeta_debito_transacciones_lag1","mactivos_margen_lag1/ctarjeta_debito_transacciones","mactivos_margen_lag1/cproductos_delta2","mactivos_margen_lag1/cproductos_delta1","mactivos_margen_lag1/cproductos","mactivos_margen_lag1/cprestamos_personales","mactivos_margen_lag1/cpayroll_trx_lag2","mactivos_margen_lag1/cpayroll_trx_lag1","mactivos_margen_lag1/cpayroll_trx","mactivos_margen_lag1/cextraccion_autoservicio","mactivos_margen_lag1/ccomisiones_otras_lag2","mactivos_margen_lag1/ccomisiones_otras","mactivos_margen_lag1/ccaja_ahorro","mactivos_margen_lag1","mactivos_margen","ctarjeta_visa_transacciones/ctarjeta_visa_transacciones_lag2","ctarjeta_visa_transacciones/ctarjeta_visa_transacciones_lag1","ctarjeta_visa_transacciones/ctarjeta_debito_transacciones_lag1","ctarjeta_visa_transacciones/cproductos_delta2","ctarjeta_visa_transacciones/cproductos_delta1","ctarjeta_visa_transacciones/cproductos","ctarjeta_visa_transacciones/cprestamos_personales","ctarjeta_visa_transacciones/cpayroll_trx_lag2","ctarjeta_visa_transacciones/cextraccion_autoservicio","ctarjeta_visa_transacciones/ccomisiones_otras_lag2","ctarjeta_visa_transacciones/ccomisiones_otras","ctarjeta_visa_transacciones/ccaja_ahorro","ctarjeta_visa_transacciones_lag2/ctarjeta_visa_transacciones_lag1","ctarjeta_visa_transacciones_lag2/ctarjeta_visa_transacciones","ctarjeta_visa_transacciones_lag2/ctarjeta_debito_transacciones_lag1","ctarjeta_visa_transacciones_lag2/ctarjeta_debito_transacciones","ctarjeta_visa_transacciones_lag2/cproductos_delta2","ctarjeta_visa_transacciones_lag2/cproductos_delta1","ctarjeta_visa_transacciones_lag2/cproductos","ctarjeta_visa_transacciones_lag2/cprestamos_personales","ctarjeta_visa_transacciones_lag2/cpayroll_trx_lag2","ctarjeta_visa_transacciones_lag2/cpayroll_trx_lag1","ctarjeta_visa_transacciones_lag2/cpayroll_trx","ctarjeta_visa_transacciones_lag2/cextraccion_autoservicio","ctarjeta_visa_transacciones_lag2/ccomisiones_otras_lag2","ctarjeta_visa_transacciones_lag2/ccomisiones_otras","ctarjeta_visa_transacciones_lag2/ccaja_ahorro","ctarjeta_visa_transacciones_lag2","ctarjeta_visa_transacciones_lag1/ctarjeta_visa_transacciones_lag2","ctarjeta_visa_transacciones_lag1/ctarjeta_visa_transacciones","ctarjeta_visa_transacciones_lag1/ctarjeta_debito_transacciones_lag1","ctarjeta_visa_transacciones_lag1/ctarjeta_debito_transacciones","ctarjeta_visa_transacciones_lag1/cproductos_delta2","ctarjeta_visa_transacciones_lag1/cproductos_delta1","ctarjeta_visa_transacciones_lag1/cproductos","ctarjeta_visa_transacciones_lag1/cprestamos_personales","ctarjeta_visa_transacciones_lag1/cpayroll_trx_lag2")

campos_malos11 <- c("cproductos_delta2/ctarjeta_visa_transacciones_lag1","cproductos_delta2/ctarjeta_visa_transacciones","cproductos_delta2/ctarjeta_debito_transacciones_lag1","cproductos_delta2/ctarjeta_debito_transacciones","cproductos_delta2/cproductos_delta1","cproductos_delta2/cproductos","cproductos_delta2/cpayroll_trx_lag2","cproductos_delta2/cpayroll_trx_lag1","cproductos_delta2/cextraccion_autoservicio","cproductos_delta2/ccomisiones_otras_lag2","cproductos_delta2/ccomisiones_otras","cproductos_delta2/ccaja_ahorro","cproductos_delta2","cproductos_delta1/ctarjeta_visa_transacciones_lag2","cproductos_delta1/ctarjeta_visa_transacciones_lag1","cproductos_delta1/ctarjeta_visa_transacciones","cproductos_delta1/ctarjeta_debito_transacciones_lag1","cproductos_delta1/ctarjeta_debito_transacciones","cproductos_delta1/cproductos_delta2","cproductos_delta1/cproductos","cproductos_delta1/cprestamos_personales","cproductos_delta1/cpayroll_trx_lag2","cproductos_delta1/cpayroll_trx_lag1","cproductos_delta1/cpayroll_trx","cproductos_delta1/cextraccion_autoservicio","cproductos_delta1/ccomisiones_otras_lag2","cproductos_delta1/ccomisiones_otras","cproductos_delta1/ccaja_ahorro","cproductos_delta1","cprestamos_personales/ctarjeta_visa_transacciones_lag2","cprestamos_personales/ctarjeta_visa_transacciones_lag1","cprestamos_personales/ctarjeta_visa_transacciones","cprestamos_personales/ctarjeta_debito_transacciones_lag1","cprestamos_personales/ctarjeta_debito_transacciones","cprestamos_personales/cproductos_delta2","cprestamos_personales/cproductos_delta1","cprestamos_personales/cproductos","cprestamos_personales/cpayroll_trx_lag2","cprestamos_personales/cpayroll_trx_lag1","cprestamos_personales/cpayroll_trx","cprestamos_personales/cextraccion_autoservicio","cprestamos_personales/ccomisiones_otras_lag2","cprestamos_personales/ccomisiones_otras","cprestamos_personales/ccaja_ahorro","cprestamos_personales_tend1","cprestamos_personales","cpayroll_trx/ctarjeta_visa_transacciones_lag2","cpayroll_trx/ctarjeta_visa_transacciones_lag1","cpayroll_trx/ctarjeta_debito_transacciones_lag1","cpayroll_trx/cproductos_delta2","cpayroll_trx/cproductos_delta1","cpayroll_trx/cproductos","cpayroll_trx/cpayroll_trx_lag2","cpayroll_trx/cpayroll_trx_lag1","cpayroll_trx/cextraccion_autoservicio","cpayroll_trx/ccomisiones_otras_lag2","cpayroll_trx/ccomisiones_otras","cpayroll_trx_tend1","cpayroll_trx_lag2/ctarjeta_visa_transacciones_lag2","cpayroll_trx_lag2/ctarjeta_visa_transacciones_lag1")

campos_malos12 <- c("cpayroll_trx_lag2/ctarjeta_visa_transacciones","cpayroll_trx_lag2/ctarjeta_debito_transacciones_lag1","cpayroll_trx_lag2/ctarjeta_debito_transacciones","cpayroll_trx_lag2/cproductos_delta2","cpayroll_trx_lag2/cproductos_delta1","cpayroll_trx_lag2/cproductos","cpayroll_trx_lag2/cprestamos_personales","cpayroll_trx_lag2/cpayroll_trx_lag1","cpayroll_trx_lag2/cpayroll_trx","cpayroll_trx_lag2/cextraccion_autoservicio","cpayroll_trx_lag2/ccomisiones_otras_lag2","cpayroll_trx_lag2/ccomisiones_otras","cpayroll_trx_lag2/ccaja_ahorro","cpayroll_trx_lag2","cpayroll_trx_lag1/ctarjeta_visa_transacciones_lag2","cpayroll_trx_lag1/ctarjeta_visa_transacciones_lag1","cpayroll_trx_lag1/ctarjeta_visa_transacciones","cpayroll_trx_lag1/ctarjeta_debito_transacciones_lag1","cpayroll_trx_lag1/cproductos_delta2","cpayroll_trx_lag1/cproductos_delta1","cpayroll_trx_lag1/cproductos","cpayroll_trx_lag1/cprestamos_personales","cpayroll_trx_lag1/cpayroll_trx_lag2","cpayroll_trx_lag1/cpayroll_trx","cpayroll_trx_lag1/cextraccion_autoservicio","cpayroll_trx_lag1/ccomisiones_otras_lag2","cpayroll_trx_lag1/ccomisiones_otras","cpayroll_trx_lag1/ccaja_ahorro","cpayroll_trx_lag1","cpayroll_trx","cpagodeservicios","cliente_antiguedad_lag2","cliente_antiguedad_lag1","cliente_antiguedad","chomebanking_transacciones_lag2","chomebanking_transacciones_lag1","cextraccion_autoservicio/ctarjeta_visa_transacciones_lag2","cextraccion_autoservicio/ctarjeta_visa_transacciones_lag1","cextraccion_autoservicio/ctarjeta_debito_transacciones_lag1","cextraccion_autoservicio/ctarjeta_debito_transacciones","cextraccion_autoservicio/cproductos_delta2","cextraccion_autoservicio/cproductos_delta1","cextraccion_autoservicio/cproductos","cextraccion_autoservicio/cprestamos_personales","cextraccion_autoservicio/cpayroll_trx_lag2","cextraccion_autoservicio/cpayroll_trx_lag1","cextraccion_autoservicio/cpayroll_trx","cextraccion_autoservicio/ccomisiones_otras_lag2","cextraccion_autoservicio/ccomisiones_otras","cextraccion_autoservicio/ccaja_ahorro","cextraccion_autoservicio_tend1","cextraccion_autoservicio","ccomisiones_otras/ctarjeta_visa_transacciones_lag2","ccomisiones_otras/ctarjeta_visa_transacciones_lag1","ccomisiones_otras/ctarjeta_visa_transacciones","ccomisiones_otras/ctarjeta_debito_transacciones_lag1","ccomisiones_otras/ctarjeta_debito_transacciones","ccomisiones_otras/cproductos_delta2","ccomisiones_otras/cproductos_delta1","ccomisiones_otras/cproductos","ccomisiones_otras/cprestamos_personales","ccomisiones_otras/cpayroll_trx_lag2","ccomisiones_otras/cpayroll_trx_lag1","ccomisiones_otras/cpayroll_trx","ccomisiones_otras/cextraccion_autoservicio","ccomisiones_otras/ccomisiones_otras_lag2","ccomisiones_otras/ccaja_ahorro","ccomisiones_otras_lag2/ctarjeta_visa_transacciones_lag2","ccomisiones_otras_lag2/ctarjeta_visa_transacciones_lag1","ccomisiones_otras_lag2/ctarjeta_visa_transacciones","ccomisiones_otras_lag2/ctarjeta_debito_transacciones_lag1","ccomisiones_otras_lag2/ctarjeta_debito_transacciones","ccomisiones_otras_lag2/cproductos_delta2","ccomisiones_otras_lag2/cproductos_delta1","ccomisiones_otras_lag2/cproductos","ccomisiones_otras_lag2/cprestamos_personales","ccomisiones_otras_lag2/cpayroll_trx_lag2","ccomisiones_otras_lag2/cpayroll_trx_lag1","ccomisiones_otras_lag2/cpayroll_trx","ccomisiones_otras_lag2/cextraccion_autoservicio","ccomisiones_otras_lag2/ccomisiones_otras","ccomisiones_otras_lag2/ccaja_ahorro","ccomisiones_otras_lag2","ccomisiones_otras","ccheques_emitidos_rechazados_delta2","ccajeros_propios_descuentos_lag1","ccajeros_propios_descuentos","ccaja_ahorro/ctarjeta_visa_transacciones_lag2","ccaja_ahorro/ctarjeta_visa_transacciones_lag1","ccaja_ahorro/ctarjeta_debito_transacciones_lag1","ccaja_ahorro/ctarjeta_debito_transacciones","ccaja_ahorro/cproductos_delta2","ccaja_ahorro/cproductos_delta1","ccaja_ahorro/cprestamos_personales","ccaja_ahorro/cpayroll_trx_lag2","ccaja_ahorro/cpayroll_trx_lag1","ccaja_ahorro/cextraccion_autoservicio","ccaja_ahorro/ccomisiones_otras_lag2","ccaja_ahorro/ccomisiones_otras","ccaja_ahorro")

campos_malos13 <- c("cproductos_delta2/ctarjeta_debito_transacciones","cproductos_delta2/ctarjeta_debito_transacciones_lag1","cproductos_delta2/ctarjeta_visa_transacciones","cproductos_delta2/ctarjeta_visa_transacciones_lag1","cproductos_delta2/ctarjeta_visa_transacciones_lag2","cproductos/ccomisiones_otras","cproductos/ccomisiones_otras_lag2","cproductos/cextraccion_autoservicio","cproductos/cpayroll_trx","cproductos/cpayroll_trx_lag1","cproductos/cpayroll_trx_lag2","cproductos/cprestamos_personales","cproductos/cproductos_delta1","cproductos/cproductos_delta2","cproductos/ctarjeta_debito_transacciones","cproductos/ctarjeta_debito_transacciones_lag1","cproductos/ctarjeta_visa_transacciones","cproductos/ctarjeta_visa_transacciones_lag1","cproductos/ctarjeta_visa_transacciones_lag2","ctarjeta_debito_transacciones","ctarjeta_debito_transacciones_lag1","ctarjeta_debito_transacciones_lag1/ccaja_ahorro","ctarjeta_debito_transacciones_lag1/ccomisiones_otras","ctarjeta_debito_transacciones_lag1/ccomisiones_otras_lag2","ctarjeta_debito_transacciones_lag1/cextraccion_autoservicio","ctarjeta_debito_transacciones_lag1/cpayroll_trx","ctarjeta_debito_transacciones_lag1/cpayroll_trx_lag1","ctarjeta_debito_transacciones_lag1/cpayroll_trx_lag2","ctarjeta_debito_transacciones_lag1/cprestamos_personales","ctarjeta_debito_transacciones_lag1/cproductos","ctarjeta_debito_transacciones_lag1/cproductos_delta1","ctarjeta_debito_transacciones_lag1/cproductos_delta2","ctarjeta_debito_transacciones_lag1/ctarjeta_debito_transacciones","ctarjeta_debito_transacciones_lag1/ctarjeta_visa_transacciones")

campos_malos14 <- c("ctarjeta_debito_transacciones_lag1/ctarjeta_visa_transacciones_lag1","ctarjeta_debito_transacciones_lag1/ctarjeta_visa_transacciones_lag2","ctarjeta_debito_transacciones_tend1","ctarjeta_debito_transacciones/ccaja_ahorro","ctarjeta_debito_transacciones/ccomisiones_otras","ctarjeta_debito_transacciones/ccomisiones_otras_lag2","ctarjeta_debito_transacciones/cpayroll_trx_lag1","ctarjeta_debito_transacciones/cpayroll_trx_lag2","ctarjeta_debito_transacciones/cprestamos_personales","ctarjeta_debito_transacciones/cproductos","ctarjeta_debito_transacciones/cproductos_delta1","ctarjeta_debito_transacciones/cproductos_delta2","ctarjeta_debito_transacciones/ctarjeta_debito_transacciones_lag1","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones_lag1","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones_lag2","ctarjeta_master_descuentos","ctarjeta_master_descuentos_delta1","ctarjeta_master_descuentos_lag1","ctarjeta_visa_transacciones_lag1","ctarjeta_visa_transacciones_lag1/ccaja_ahorro","ctarjeta_visa_transacciones_lag1/ccomisiones_otras","ctarjeta_visa_transacciones_lag1/ccomisiones_otras_lag2","ctarjeta_visa_transacciones_lag1/cextraccion_autoservicio","ctarjeta_visa_transacciones_lag1/cpayroll_trx","ctarjeta_visa_transacciones_lag1/cpayroll_trx_lag1","ctarjeta_visa_transacciones_lag1/cpayroll_trx_lag2","ctarjeta_visa_transacciones_lag1/cprestamos_personales","ctarjeta_visa_transacciones_lag1/cproductos","ctarjeta_visa_transacciones_lag1/cproductos_delta1","ctarjeta_visa_transacciones_lag1/cproductos_delta2","ctarjeta_visa_transacciones_lag1/ctarjeta_debito_transacciones","ctarjeta_visa_transacciones_lag1/ctarjeta_debito_transacciones_lag1","ctarjeta_visa_transacciones_lag1/ctarjeta_visa_transacciones","ctarjeta_visa_transacciones_lag1/ctarjeta_visa_transacciones_lag2","ctarjeta_visa_transacciones_lag2","ctarjeta_visa_transacciones_lag2/ccaja_ahorro","ctarjeta_visa_transacciones_lag2/ccomisiones_otras","ctarjeta_visa_transacciones_lag2/ccomisiones_otras_lag2","ctarjeta_visa_transacciones_lag2/cextraccion_autoservicio","ctarjeta_visa_transacciones_lag2/cpayroll_trx","ctarjeta_visa_transacciones_lag2/cpayroll_trx_lag1","ctarjeta_visa_transacciones_lag2/cpayroll_trx_lag2","ctarjeta_visa_transacciones_lag2/cprestamos_personales","ctarjeta_visa_transacciones_lag2/cproductos","ctarjeta_visa_transacciones_lag2/cproductos_delta1","ctarjeta_visa_transacciones_lag2/cproductos_delta2","ctarjeta_visa_transacciones_lag2/ctarjeta_debito_transacciones","ctarjeta_visa_transacciones_lag2/ctarjeta_debito_transacciones_lag1","ctarjeta_visa_transacciones_lag2/ctarjeta_visa_transacciones","ctarjeta_visa_transacciones_lag2/ctarjeta_visa_transacciones_lag1","ctarjeta_visa_transacciones/ccaja_ahorro")

campos_malos15 <- c("mdescubierto_preacordado/ccomisiones_otras_lag2","mdescubierto_preacordado/cextraccion_autoservicio","mdescubierto_preacordado/cpayroll_trx_lag1","mdescubierto_preacordado/cpayroll_trx_lag2","mdescubierto_preacordado/cprestamos_personales","mdescubierto_preacordado/cproductos_delta1","mdescubierto_preacordado/cproductos_delta2","mdescubierto_preacordado/ctarjeta_debito_transacciones","mdescubierto_preacordado/ctarjeta_debito_transacciones_lag1","mdescubierto_preacordado/ctarjeta_visa_transacciones","mdescubierto_preacordado/ctarjeta_visa_transacciones_lag1","mdescubierto_preacordado/ctarjeta_visa_transacciones_lag2","minversion1_dolares_lag1","minversion1_dolares_lag2","mpagodeservicios","mpagodeservicios_delta1","mpagodeservicios_delta2","mpagodeservicios_lag1","mpayroll","mpayroll_delta1","mpayroll_delta2","mpayroll_lag1","mpayroll_lag2","mpayroll_tend1","mpayroll_tend1/ccaja_ahorro","mpayroll_tend1/ccomisiones_otras","mpayroll_tend1/ccomisiones_otras_lag2","mpayroll_tend1/cextraccion_autoservicio","mpayroll_tend1/cpayroll_trx","mpayroll_tend1/cpayroll_trx_lag1","mpayroll_tend1/cpayroll_trx_lag2","mpayroll_tend1/cprestamos_personales","mpayroll_tend1/cproductos","mpayroll_tend1/cproductos_delta1","mpayroll_tend1/cproductos_delta2","mpayroll_tend1/ctarjeta_debito_transacciones","mpayroll_tend1/ctarjeta_debito_transacciones_lag1","mpayroll_tend1/ctarjeta_visa_transacciones","mpayroll_tend1/ctarjeta_visa_transacciones_lag1","mpayroll_tend1/ctarjeta_visa_transacciones_lag2","mpayroll2_tend1","mplazo_fijo_pesos_delta2","mprestamos_personales","mprestamos_personales_delta1","mprestamos_personales_lag1","mprestamos_personales_lag2","mprestamos_personales_lag2/ccomisiones_otras","mprestamos_personales_lag2/ccomisiones_otras_lag2","mprestamos_personales_lag2/cextraccion_autoservicio")

campos_malos16 <- c("mv_msaldopesos/cproductos_delta1","mv_msaldopesos/cproductos","mv_msaldopesos/cprestamos_personales","mv_msaldopesos/cpayroll_trx_lag2","mv_msaldopesos/cpayroll_trx_lag1","mv_msaldopesos/cpayroll_trx","mv_msaldopesos/cextraccion_autoservicio","mv_msaldopesos/ccomisiones_otras_lag2","mv_msaldopesos/ccomisiones_otras","mv_msaldopesos/ccaja_ahorro","mv_msaldopesos","mv_mpagospesos/ctarjeta_visa_transacciones_lag2","mv_mpagospesos/ctarjeta_visa_transacciones_lag1","mv_mpagospesos/ctarjeta_visa_transacciones","mv_mpagospesos/ctarjeta_debito_transacciones_lag1","mv_mpagospesos/ctarjeta_debito_transacciones","mv_mpagospesos/cproductos_delta2","mv_mpagospesos/cproductos_delta1","mv_mpagospesos/cproductos","mv_mpagospesos/cprestamos_personales","mv_mpagospesos/cpayroll_trx_lag2","mv_mpagospesos/cpayroll_trx_lag1","mv_mpagospesos/cpayroll_trx","mv_mpagospesos/cextraccion_autoservicio","mv_mpagospesos/ccomisiones_otras_lag2","mv_mpagospesos/ccomisiones_otras","mv_mpagospesos/ccaja_ahorro","mv_mpagospesos","mv_mpagominimo/ctarjeta_visa_transacciones_lag2","mv_mpagominimo/ctarjeta_visa_transacciones_lag1","mv_mpagominimo/ctarjeta_visa_transacciones","mv_mpagominimo/ctarjeta_debito_transacciones_lag1","mv_mpagominimo/ctarjeta_debito_transacciones","mv_mpagominimo/cproductos_delta2","mv_mpagominimo/cproductos_delta1","mv_mpagominimo/cproductos","mv_mpagominimo/cprestamos_personales","mv_mpagominimo/cpayroll_trx_lag2","mv_mpagominimo/cpayroll_trx_lag1","mv_mpagominimo/cextraccion_autoservicio","mv_mpagominimo/ccomisiones_otras_lag2","mv_mpagominimo/ccomisiones_otras","mv_mpagominimo/ccaja_ahorro","mv_mpagominimo","mv_mlimitecompra","mv_mfinanciacion_limite","mv_mconsumototal/ctarjeta_visa_transacciones_lag2","mv_mconsumototal/ctarjeta_visa_transacciones_lag1")

campos_malos17 <- c("ctarjeta_visa_transacciones_lag1/cpayroll_trx_lag1","ctarjeta_visa_transacciones_lag1/cpayroll_trx","ctarjeta_visa_transacciones_lag1/cextraccion_autoservicio","ctarjeta_visa_transacciones_lag1/ccomisiones_otras_lag2","ctarjeta_visa_transacciones_lag1/ccomisiones_otras","ctarjeta_visa_transacciones_lag1/ccaja_ahorro","ctarjeta_visa_transacciones_lag1","ctarjeta_master_descuentos_lag1","ctarjeta_master_descuentos_delta1","ctarjeta_master_descuentos","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones_lag2","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones_lag1","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones","ctarjeta_debito_transacciones/ctarjeta_debito_transacciones_lag1","ctarjeta_debito_transacciones/cproductos_delta2","ctarjeta_debito_transacciones/cproductos_delta1","ctarjeta_debito_transacciones/cproductos","ctarjeta_debito_transacciones/cprestamos_personales","ctarjeta_debito_transacciones/cpayroll_trx_lag2","ctarjeta_debito_transacciones/cpayroll_trx_lag1","ctarjeta_debito_transacciones/ccomisiones_otras_lag2","ctarjeta_debito_transacciones/ccomisiones_otras","ctarjeta_debito_transacciones/ccaja_ahorro","ctarjeta_debito_transacciones_tend1","ctarjeta_debito_transacciones_lag1/ctarjeta_visa_transacciones_lag2","ctarjeta_debito_transacciones_lag1/ctarjeta_visa_transacciones_lag1","ctarjeta_debito_transacciones_lag1/ctarjeta_visa_transacciones","ctarjeta_debito_transacciones_lag1/ctarjeta_debito_transacciones","ctarjeta_debito_transacciones_lag1/cproductos_delta2","ctarjeta_debito_transacciones_lag1/cproductos_delta1","ctarjeta_debito_transacciones_lag1/cproductos","ctarjeta_debito_transacciones_lag1/cprestamos_personales","ctarjeta_debito_transacciones_lag1/cpayroll_trx_lag2","ctarjeta_debito_transacciones_lag1/cpayroll_trx_lag1","ctarjeta_debito_transacciones_lag1/cpayroll_trx","ctarjeta_debito_transacciones_lag1/cextraccion_autoservicio","ctarjeta_debito_transacciones_lag1/ccomisiones_otras_lag2","ctarjeta_debito_transacciones_lag1/ccomisiones_otras","ctarjeta_debito_transacciones_lag1/ccaja_ahorro","ctarjeta_debito_transacciones_lag1","ctarjeta_debito_transacciones","cproductos/ctarjeta_visa_transacciones_lag2","cproductos/ctarjeta_visa_transacciones_lag1","cproductos/ctarjeta_visa_transacciones","cproductos/ctarjeta_debito_transacciones_lag1","cproductos/ctarjeta_debito_transacciones","cproductos/cproductos_delta2","cproductos/cproductos_delta1","cproductos/cprestamos_personales","cproductos/cpayroll_trx_lag2","cproductos/cpayroll_trx_lag1","cproductos/cpayroll_trx","cproductos/cextraccion_autoservicio","cproductos/ccomisiones_otras_lag2","cproductos/ccomisiones_otras","cproductos_delta2/ctarjeta_visa_transacciones_lag2")


ksemilla_azar  <- 102191  #Aqui poner la propia semilla
#------------------------------------------------------------------------------
#Funcion que lleva el registro de los experimentos

get_experimento  <- function()
{
  if( !file.exists( "./maestro.yaml" ) )  cat( file="./maestro.yaml", "experimento: 1000" )
  
  exp  <- read_yaml( "./maestro.yaml" )
  experimento_actual  <- exp$experimento
  
  exp$experimento  <- as.integer(exp$experimento + 1)
  Sys.chmod( "./maestro.yaml", mode = "0644", use_umask = TRUE)
  write_yaml( exp, "./maestro.yaml" )
  Sys.chmod( "./maestro.yaml", mode = "0444", use_umask = TRUE) #dejo el archivo readonly
  
  return( experimento_actual )
}
#------------------------------------------------------------------------------
#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos

loguear  <- function( reg, arch=NA, folder="./work/", ext=".txt", verbose=TRUE )
{
  archivo  <- arch
  if( is.na(arch) )  archivo  <- paste0(  folder, substitute( reg), ext )
  
  if( !file.exists( archivo ) )  #Escribo los titulos
  {
    linea  <- paste0( "fecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )
    
    cat( linea, file=archivo )
  }
  
  linea  <- paste0( format(Sys.time(), "%Y%m%d %H%M%S"),  "\t",     #la fecha y hora
                    gsub( ", ", "\t", toString( reg ) ),  "\n" )
  
  cat( linea, file=archivo, append=TRUE )  #grabo al archivo
  
  if( verbose )  cat( linea )   #imprimo por pantalla
}
#------------------------------------------------------------------------------

PROB_CORTE  <- 0.025

fganancia_logistic_lightgbm   <- function(probs, datos) 
{
  vlabels  <- getinfo(datos, "label")
  vpesos   <- getinfo(datos, "weight")
  
  #aqui esta el inmoral uso de los pesos para calcular la ganancia correcta
  gan  <- sum( (probs > PROB_CORTE  ) *
                 ifelse( vlabels== 1 & vpesos > 1, 48750, -1250 ) )
  
  return( list( "name"= "ganancia", 
                "value"=  gan,
                "higher_better"= TRUE ) )
}
#------------------------------------------------------------------------------
#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales, la semilla del mal ...

EstimarGanancia_lightgbm  <- function( x )
{
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1
  
  gc()
  PROB_CORTE <<- x$prob_corte   #asigno la variable global
  
  kfolds  <- 5   # cantidad de folds para cross validation
  
  param_basicos  <- list( objective= "binary",
                          metric= "custom",
                          first_metric_only= TRUE,
                          boost_from_average= TRUE,
                          feature_pre_filter= FALSE,
                          verbosity= -100,
                          seed= 999983,
                          max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
                          min_gain_to_split= 0.0, #por ahora, lo dejo fijo
                          lambda_l1= 0.0,         #por ahora, lo dejo fijo
                          lambda_l2= 0.0,         #por ahora, lo dejo fijo
                          max_bin= 31,            #por ahora, lo dejo fijo
                          num_iterations= 9999,    #un numero muy grande, lo limita early_stopping_rounds
                          force_row_wise= TRUE    #para que los alumnos no se atemoricen con tantos warning
  )
  
  #el parametro discolo, que depende de otro
  param_variable  <- list(  early_stopping_rounds= as.integer(50 + 1/x$learning_rate) )
  
  param_completo  <- c( param_basicos, param_variable, x )
  
  set.seed( 999983 )
  modelocv  <- lgb.cv( data= dtrain,
                       eval= fganancia_logistic_lightgbm,
                       stratified= TRUE, #sobre el cross validation
                       nfold= kfolds,    #folds del cross validation
                       param= param_completo,
                       verbose= -100
  )
  
  
  ganancia  <- unlist(modelocv$record_evals$valid$ganancia$eval)[ modelocv$best_iter ]
  
  ganancia_normalizada  <-  ganancia* kfolds  
  attr(ganancia_normalizada ,"extras" )  <- list("num_iterations"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra
  
  param_completo$num_iterations <- modelocv$best_iter  #asigno el mejor num_iterations
  param_completo["early_stopping_rounds"]  <- NULL
  
  #si tengo una ganancia superadora, genero el archivo para Kaggle
  if(  ganancia > GLOBAL_ganancia_max )
  {
    GLOBAL_ganancia_max  <<- ganancia  #asigno la nueva maxima ganancia a una variable GLOBAL, por eso el <<-
    
    set.seed(ksemilla_azar)
    
    modelo  <- lightgbm( data= dtrain,
                         param= param_completo,
                         verbose= -100
    )
    
    #calculo la importancia de variables
    tb_importancia  <- lgb.importance( model= modelo )
    fwrite( tb_importancia, 
            file= paste0(kimp, "imp_", GLOBAL_iteracion, ".txt"),
            sep="\t" )
    
    prediccion  <- predict( modelo, data.matrix( dapply[  , campos_buenos, with=FALSE]) )
    
    Predicted  <- as.integer( prediccion > x$prob_corte )
    
    entrega  <- as.data.table( list( "numero_de_cliente"= dapply$numero_de_cliente, 
                                     "Predicted"= Predicted)  )
    
    #genero el archivo para Kaggle
    fwrite( entrega, 
            file= paste0(kkaggle, GLOBAL_iteracion, ".csv" ),
            sep= "," )
  }
  
  #logueo 
  xx  <- param_completo
  xx$iteracion_bayesiana  <- GLOBAL_iteracion
  xx$ganancia  <- ganancia_normalizada   #le agrego la ganancia
  loguear( xx,  arch= klog )
  
  return( ganancia )
}
#------------------------------------------------------------------------------
#Aqui empieza el programa

if( is.na(kexperimento ) )   kexperimento <- get_experimento()  #creo el experimento

#en estos archivos quedan los resultados
dir.create( paste0( "./work/E",  kexperimento, "/" ) )
kbayesiana  <- paste0("./work/E",  kexperimento, "/E",  kexperimento, "_", kscript, ".RDATA" )
klog        <- paste0("./work/E",  kexperimento, "/E",  kexperimento, "_", kscript, "_BOlog.txt" )
kimp        <- paste0("./work/E",  kexperimento, "/E",  kexperimento, "_", kscript, "_" )
kkaggle     <- paste0("./kaggle/E",kexperimento, "_", kscript, "_" )


GLOBAL_ganancia_max  <-  -Inf
GLOBAL_iteracion  <- 0

#si ya existe el archivo log, traigo hasta donde llegue
if( file.exists(klog) )
{
  tabla_log  <- fread( klog)
  GLOBAL_iteracion  <- nrow( tabla_log ) -1
  GLOBAL_ganancia_max  <- tabla_log[ , max(ganancia) ]
}


#cargo el dataset que tiene los 36 meses
dataset  <- fread(karch_dataset)


campos_lags  <- setdiff(  colnames(dataset) ,  c("clase_ternaria","clase01", "numero_de_cliente","foto_mes", campos_malos, campos_malos2, campos_malos2bis,campos_malos2bisbisbis,campos_malosbisbis, 
                                                 campos_malos3,campos_malos3bis, campos_malos3bisbis,campos_malos4, campos_malos5,campos_malos6, campos_malos7, 
                                                 campos_malos8,campos_malos8bis,campos_malos8bisbis, campos_malos9,campos_malos9bis, campos_malos9bisbis,
                                                 campos_malos10,campos_malos11, campos_malos12, campos_malos13, campos_malos14, campos_malos15, campos_malos16,campos_malos17))

#Hago feature Engineering en este mismo script

#Saco junio

dataset <- dataset[foto_mes!=202006,]


#primero hago el FE visto en clases

#acomodo los errores del dataset menos junio que es eliminado
dataset[ foto_mes==201801,  internet   := NA ]
dataset[ foto_mes==201801,  thomebanking   := NA ]
dataset[ foto_mes==201801,  chomebanking_transacciones   := NA ]
dataset[ foto_mes==201801,  tcallcenter   := NA ]
dataset[ foto_mes==201801,  ccallcenter_transacciones   := NA ]
dataset[ foto_mes==201801,  cprestamos_personales   := NA ]
dataset[ foto_mes==201801,  mprestamos_personales   := NA ]
dataset[ foto_mes==201801,  mprestamos_hipotecarios  := NA ]
dataset[ foto_mes==201801,  ccajas_transacciones   := NA ]
dataset[ foto_mes==201801,  ccajas_consultas   := NA ]
dataset[ foto_mes==201801,  ccajas_depositos   := NA ]
dataset[ foto_mes==201801,  ccajas_extracciones   := NA ]
dataset[ foto_mes==201801,  ccajas_otras   := NA ]

dataset[ foto_mes==201806,  tcallcenter   :=  NA ]
dataset[ foto_mes==201806,  ccallcenter_transacciones   :=  NA ]

dataset[ foto_mes==201904,  ctarjeta_visa_debitos_automaticos  :=  NA ]
dataset[ foto_mes==201904,  mttarjeta_visa_debitos_automaticos := NA ]
dataset[ foto_mes==201904,  Visa_mfinanciacion_limite := NA ]

dataset[ foto_mes==201905,  mrentabilidad     := NA ]
dataset[ foto_mes==201905,  mrentabilidad_annual     := NA ]
dataset[ foto_mes==201905,  mcomisiones      := NA ]
dataset[ foto_mes==201905,  mpasivos_margen  := NA ]
dataset[ foto_mes==201905,  mactivos_margen  := NA ]
dataset[ foto_mes==201905,  ctarjeta_visa_debitos_automaticos  := NA ]
dataset[ foto_mes==201905,  ccomisiones_otras := NA ]
dataset[ foto_mes==201905,  mcomisiones_otras := NA ]

dataset[ foto_mes==201910,  mpasivos_margen   := NA ]
dataset[ foto_mes==201910,  mactivos_margen   := NA ]
dataset[ foto_mes==201910,  ccomisiones_otras := NA ]
dataset[ foto_mes==201910,  mcomisiones_otras := NA ]
dataset[ foto_mes==201910,  mcomisiones       := NA ]
dataset[ foto_mes==201910,  mrentabilidad     := NA ]
dataset[ foto_mes==201910,  mrentabilidad_annual        := NA ]
dataset[ foto_mes==201910,  chomebanking_transacciones  := NA ]
dataset[ foto_mes==201910,  ctarjeta_visa_descuentos    := NA ]
dataset[ foto_mes==201910,  ctarjeta_master_descuentos  := NA ]
dataset[ foto_mes==201910,  mtarjeta_visa_descuentos    := NA ]
dataset[ foto_mes==201910,  mtarjeta_master_descuentos  := NA ]
dataset[ foto_mes==201910,  ccajeros_propios_descuentos := NA ]
dataset[ foto_mes==201910,  mcajeros_propios_descuentos := NA ]

dataset[ foto_mes==202001,  cliente_vip   := NA ]

dataset[ foto_mes==202010,  internet  := NA ]
dataset[ foto_mes==202011,  internet  := NA ]
dataset[ foto_mes==202012,  internet  := NA ]
dataset[ foto_mes==202101,  internet  := NA ]

dataset[ foto_mes==202009,  tmobile_app  := NA ]
dataset[ foto_mes==202010,  tmobile_app  := NA ]
dataset[ foto_mes==202011,  tmobile_app  := NA ]
dataset[ foto_mes==202012,  tmobile_app  := NA ]
dataset[ foto_mes==202101,  tmobile_app  := NA ]

#agrego las variables vistas en 04.03 - FE Completo
##################INICIO MIEDO#####################

dataset[ , mv_status01       := pmax( Master_status,  Visa_status, na.rm = TRUE) ]
dataset[ , mv_status02       := Master_status +  Visa_status ]
dataset[ , mv_status03       := pmax( ifelse( is.na(Master_status), 10, Master_status) , ifelse( is.na(Visa_status), 10, Visa_status) ) ]
dataset[ , mv_status04       := ifelse( is.na(Master_status), 10, Master_status)  +  ifelse( is.na(Visa_status), 10, Visa_status)  ]
dataset[ , mv_status05       := ifelse( is.na(Master_status), 10, Master_status)  +  100*ifelse( is.na(Visa_status), 10, Visa_status)  ]

dataset[ , mv_status06       := ifelse( is.na(Visa_status), 
                                        ifelse( is.na(Master_status), 10, Master_status), 
                                        Visa_status)  ]

dataset[ , mv_status07       := ifelse( is.na(Master_status), 
                                        ifelse( is.na(Visa_status), 10, Visa_status), 
                                        Master_status)  ]


#combino MasterCard y Visa
dataset[ , mv_mfinanciacion_limite := rowSums( cbind( Master_mfinanciacion_limite,  Visa_mfinanciacion_limite) , na.rm=TRUE ) ]

dataset[ , mv_Fvencimiento         := pmin( Master_Fvencimiento, Visa_Fvencimiento, na.rm = TRUE) ]
dataset[ , mv_Finiciomora          := pmin( Master_Finiciomora, Visa_Finiciomora, na.rm = TRUE) ]
dataset[ , mv_msaldototal          := rowSums( cbind( Master_msaldototal,  Visa_msaldototal) , na.rm=TRUE ) ]
dataset[ , mv_msaldopesos          := rowSums( cbind( Master_msaldopesos,  Visa_msaldopesos) , na.rm=TRUE ) ]
dataset[ , mv_msaldodolares        := rowSums( cbind( Master_msaldodolares,  Visa_msaldodolares) , na.rm=TRUE ) ]
dataset[ , mv_mconsumospesos       := rowSums( cbind( Master_mconsumospesos,  Visa_mconsumospesos) , na.rm=TRUE ) ]
dataset[ , mv_mconsumosdolares     := rowSums( cbind( Master_mconsumosdolares,  Visa_mconsumosdolares) , na.rm=TRUE ) ]
dataset[ , mv_mlimitecompra        := rowSums( cbind( Master_mlimitecompra,  Visa_mlimitecompra) , na.rm=TRUE ) ]
dataset[ , mv_madelantopesos       := rowSums( cbind( Master_madelantopesos,  Visa_madelantopesos) , na.rm=TRUE ) ]
dataset[ , mv_madelantodolares     := rowSums( cbind( Master_madelantodolares,  Visa_madelantodolares) , na.rm=TRUE ) ]
dataset[ , mv_fultimo_cierre       := pmax( Master_fultimo_cierre, Visa_fultimo_cierre, na.rm = TRUE) ]
dataset[ , mv_mpagado              := rowSums( cbind( Master_mpagado,  Visa_mpagado) , na.rm=TRUE ) ]
dataset[ , mv_mpagospesos          := rowSums( cbind( Master_mpagospesos,  Visa_mpagospesos) , na.rm=TRUE ) ]
dataset[ , mv_mpagosdolares        := rowSums( cbind( Master_mpagosdolares,  Visa_mpagosdolares) , na.rm=TRUE ) ]
dataset[ , mv_fechaalta            := pmax( Master_fechaalta, Visa_fechaalta, na.rm = TRUE) ]
dataset[ , mv_mconsumototal        := rowSums( cbind( Master_mconsumototal,  Visa_mconsumototal) , na.rm=TRUE ) ]
dataset[ , mv_cconsumos            := rowSums( cbind( Master_cconsumos,  Visa_cconsumos) , na.rm=TRUE ) ]
dataset[ , mv_cadelantosefectivo   := rowSums( cbind( Master_cadelantosefectivo,  Visa_cadelantosefectivo) , na.rm=TRUE ) ]
dataset[ , mv_mpagominimo          := rowSums( cbind( Master_mpagominimo,  Visa_mpagominimo) , na.rm=TRUE ) ]

#a partir de aqui juego con la suma de Mastercard y Visa
dataset[ , mvr_Master_mlimitecompra:= Master_mlimitecompra / mv_mlimitecompra ]
dataset[ , mvr_Visa_mlimitecompra  := Visa_mlimitecompra / mv_mlimitecompra ]
dataset[ , mvr_msaldototal         := mv_msaldototal / mv_mlimitecompra ]
dataset[ , mvr_msaldopesos         := mv_msaldopesos / mv_mlimitecompra ]
dataset[ , mvr_msaldopesos2        := mv_msaldopesos / mv_msaldototal ]
dataset[ , mvr_msaldodolares       := mv_msaldodolares / mv_mlimitecompra ]
dataset[ , mvr_msaldodolares2      := mv_msaldodolares / mv_msaldototal ]
dataset[ , mvr_mconsumospesos      := mv_mconsumospesos / mv_mlimitecompra ]
dataset[ , mvr_mconsumosdolares    := mv_mconsumosdolares / mv_mlimitecompra ]
dataset[ , mvr_madelantopesos      := mv_madelantopesos / mv_mlimitecompra ]
dataset[ , mvr_madelantodolares    := mv_madelantodolares / mv_mlimitecompra ]
dataset[ , mvr_mpagado             := mv_mpagado / mv_mlimitecompra ]
dataset[ , mvr_mpagospesos         := mv_mpagospesos / mv_mlimitecompra ]
dataset[ , mvr_mpagosdolares       := mv_mpagosdolares / mv_mlimitecompra ]
dataset[ , mvr_mconsumototal       := mv_mconsumototal  / mv_mlimitecompra ]
dataset[ , mvr_mpagominimo         := mv_mpagominimo  / mv_mlimitecompra ]

#En esta seccion juego con indices, cocientes, etc de variables ya detectadas como relevantes

######################COMPLETAR#########################

#valvula de seguridad para evitar valores infinitos
#paso los infinitos a NULOS
infinitos      <- lapply(names(dataset),function(.name) dataset[ , sum(is.infinite(get(.name)))])
infinitos_qty  <- sum( unlist( infinitos) )
if( infinitos_qty > 0 )
{
  cat( "ATENCION, hay", infinitos_qty, "valores infinitos en tu dataset. Seran pasados a NA\n" )
  dataset[mapply(is.infinite, dataset)] <- NA
}


#valvula de seguridad para evitar valores NaN  que es 0/0
#paso los NaN a 0 , decision polemica si las hay
#se invita a asignar un valor razonable segun la semantica del campo creado
nans      <- lapply(names(dataset),function(.name) dataset[ , sum(is.nan(get(.name)))])
nans_qty  <- sum( unlist( nans) )
if( nans_qty > 0 )
{
  cat( "ATENCION, hay", nans_qty, "valores NaN 0/0 en tu dataset. Seran pasados arbitrariamente a 0\n" )
  cat( "Si no te gusta la decision, modifica a gusto el programa!\n\n")
  dataset[mapply(is.nan, dataset)] <- 0
}

##################FIN MIEDO#####################

#agreglo los lags de orden 1,2,3 y 4
setorderv( dataset, c("numero_de_cliente","foto_mes") )
dataset[ , paste0( campos_lags, "_lag1") := shift(.SD, 1, NA, "lag"), 
         by= numero_de_cliente, 
         .SDcols= campos_lags]

dataset[ , paste0( campos_lags, "_lag2") := shift(.SD, 2, NA, "lag"), 
         by= numero_de_cliente, 
         .SDcols= campos_lags]

dataset[ , paste0( campos_lags, "_lag3") := shift(.SD, 3, NA, "lag"), 
         by= numero_de_cliente, 
         .SDcols= campos_lags]

dataset[ , paste0( campos_lags, "_lag4") := shift(.SD, 4, NA, "lag"), 
         by= numero_de_cliente, 
         .SDcols= campos_lags]

for(i in range(1:10))
  
  #agrego los deltas de los lags, con un "for" nada elegante
  for( vcol in campos_lags )
  {
    dataset[,  paste0(vcol, "_delta1") := get( vcol)  - get(paste0( vcol, "_lag1"))]
  }

for( vcol in campos_lags ){
  dataset[,  paste0(vcol, "_delta2") := get( vcol)  - get(paste0( vcol, "_lag2"))]
}

for( vcol in campos_lags ){
  dataset[,  paste0(vcol, "_tend1") := (get( vcol)+get(paste0( vcol, "_lag1")))/(get(paste0( vcol, "_lag2"))+get(paste0( vcol, "_lag3"))+get(paste0( vcol, "_lag4"))) ]
}
list_drop_3 = c(paste0( campos_lags, "_lag3"))
list_drop_4 = c(paste0( campos_lags, "_lag4"))
dataset[, c(list_drop_3):=NULL]  # remove columns
dataset[, c(list_drop_4):=NULL]  # remove columns

###########esto no se crea porque no encuentra las variables########
#creo vector con variables flash 

tipom <- c("mtarjeta_visa_consumo","mprestamos_personales","mv_msaldototal","Visa_mpagominimo","mtarjeta_visa_consumo_lag1","mv_status04","mactivos_margen","mv_msaldopesos","mcuenta_corriente","mvr_msaldopesos","mv_mpagospesos","mvr_msaldototal","Visa_msaldototal","mdescubierto_preacordado","mtransferencias_recibidas_lag1","mcuenta_corriente_lag1","mrentabilidad","mrentabilidad_annual_lag2","mrentabilidad_lag1","mactivos_margen_lag1","mrentabilidad_tend1","mv_mconsumototal","mrentabilidad_annual","mv_mpagominimo","mvr_mpagospesos","mprestamos_personales_lag2","mrentabilidad_lag2","mv_mconsumospesos","mcaja_ahorro_dolares","mactivos_margen_lag2","Master_mlimitecompra_lag2","mpayroll_tend1","Master_fechaalta_lag2","mcomisiones_otras","Visa_mlimitecompra","Visa_msaldototal_delta1","mcomisiones_lag2","Visa_msaldototal_lag1","Visa_mconsumototal_lag1","mpayroll","mpayroll_lag1","mtransferencias_recibidas")
tipoc <- c("ctarjeta_visa_transacciones","cpayroll_trx","cpayroll_trx_lag2","ccomisiones_otras","cpayroll_trx_lag1","ctarjeta_debito_transacciones","ctarjeta_debito_transacciones_lag1","cproductos_delta1","cextraccion_autoservicio","ctarjeta_visa_transacciones_lag1","ccaja_ahorro","ctarjeta_visa_transacciones_lag2","cproductos_delta2","cproductos","ccomisiones_otras_lag2","cprestamos_personales")

#creo cocientes entre tipom y tipoc

for (vcol in tipom){
  for (vcol2 in tipoc) {
    dataset[, paste0(vcol, "/",vcol2) := get(vcol)/get(vcol2)]
  }
}

#creo cocientes entre tipom y tipom

for (vcol in tipom){
  for (vcol2 in tipom) {
    if(vcol != vcol2){
      dataset[, paste0(vcol, "/",vcol2) := get(vcol)/get(vcol2)]
    }
  }
}

#creo cocientes entre tipoc y tipoc

for (vcol in tipoc){
  for (vcol2 in tipoc) {
    if(vcol != vcol2){
      dataset[, paste0(vcol, "/",vcol2) := get(vcol)/get(vcol2)]
    }
  }
}

#agrego canaritos
if( kcanaritos > 0 )
{
  for( i  in 1:kcanaritos)  dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset))]
}

#cargo los datos donde voy a aplicar el modelo
dapply  <- copy( dataset[  foto_mes==kmes_apply ] )


#creo la clase_binaria2   1={ BAJA+2,BAJA+1}  0={CONTINUA}
dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]



#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01", "numero_de_cliente","foto_mes", campos_malos, campos_malos2, campos_malos2bis,campos_malos2bisbisbis,campos_malosbisbis, 
                                                campos_malos3,campos_malos3bis, campos_malos3bisbis,campos_malos4, campos_malos5,campos_malos6, campos_malos7, 
                                                campos_malos8,campos_malos8bis,campos_malos8bisbis, campos_malos9,campos_malos9bis, campos_malos9bisbis,
                                                campos_malos10,campos_malos11, campos_malos12, campos_malos13, campos_malos14, campos_malos15, campos_malos16,campos_malos17))


##############################################################
#despues de esto es mucho tiempo#

#dejo los datos en el formato que necesita LightGBM
#uso el weight como un truco ESPANTOSO para saber la clase real
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[  foto_mes>=kmes_train_desde & foto_mes<=kmes_train_hasta , campos_buenos, with=FALSE]),
                        label=  dataset[ foto_mes>=kmes_train_desde & foto_mes<=kmes_train_hasta, clase01],
                        weight=  dataset[ foto_mes>=kmes_train_desde & foto_mes<=kmes_train_hasta , ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)] ,
                        free_raw_data= TRUE
)

#elimino el dataset para liberar memoria RAM
rm( dataset )
gc()

#Aqui comienza la configuracion de la Bayesian Optimization

funcion_optimizar  <- EstimarGanancia_lightgbm   #la funcion que voy a maximizar

configureMlr( show.learner.output= FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo
obj.fun  <- makeSingleObjectiveFunction(
  fn=       funcion_optimizar, #la funcion que voy a maximizar
  minimize= FALSE,   #estoy Maximizando la ganancia
  noisy=    TRUE,
  par.set=  hs,     #definido al comienzo del programa
  has.simple.signature = FALSE   #paso los parametros en una lista
)

ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)  #se graba cada 600 segundos
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )   #cantidad de iteraciones
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )

#establezco la funcion que busca el maximo
surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))

#inicio la optimizacion bayesiana
if(!file.exists(kbayesiana)) {
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl)
} else {
  run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista
}



#apagado de la maquina virtual, pero NO se borra
system( "sleep 10  &&  sudo shutdown -h now", wait=FALSE)

#suicidio,  elimina la maquina virtual directamente
#system( "sleep 10  && 
#        export NAME=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google') &&
#        export ZONE=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google') &&
#        gcloud --quiet compute instances delete $NAME --zone=$ZONE",
#        wait=FALSE )


quit( save="no" )
