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

kscript         <- "lgb_bin2_lag1&2_elimcanar_FE_eliminarvarextras_cocientesdevariablesvarias"

karch_dataset    <- "./datasetsOri/paquete_premium.csv.gz"
kmes_apply       <- 202101  #El mes donde debo aplicar el modelo
kmes_train_hasta <- 202011  #Obvimente, solo puedo entrenar hasta 202011

kmes_train_desde <- 202001  #Entreno desde Enero-2020

kcanaritos  <-  100
kBO_iter    <-  100   #cantidad de iteraciones de la Optimizacion Bayesiana

#Aqui se cargan los hiperparametros
hs <- makeParamSet( 
  makeNumericParam("learning_rate",    lower=    0.01 , upper=    0.05),
  makeNumericParam("feature_fraction", lower=    0.1  , upper=    0.4), 
  makeIntegerParam("min_data_in_leaf", lower= 2000L   , upper= 4000L), 
  makeIntegerParam("num_leaves",       lower=  100L   , upper= 1024L), 
  makeNumericParam("prob_corte",       lower=    0.020, upper=    0.050)
)

campos_malos  <- c("mpasivos_margen","mcuentas_saldo","mautoservicio", "cpagomiscuentas", "mpagomiscuentas", "mtarjeta_visa_descuentos", "ctrx_quarter", "Master_mfinanciacion_limite", "Master_mconsumospesos", "Master_fultimo_cierre","Master_mpagominimo","Visa_mfinanciacion_limite","Visa_msaldopesos","Visa_mconsumospesos", "Visa_fultimo_cierre", "Visa_mpagospesos")   #aqui se deben cargar todos los campos culpables del Data Drifting

#aca van campos que no suman al arbol
campos_malos2 <- c('mcheques_depositados_rechazados_tend1','ccajeros_propios_descuentos_delta1','mtarjeta_master_descuentos','cpagodeservicios_delta1','cinversion1_delta2','cseguro_auto_lag2','tpaquete1_delta2','Master_madelantopesos_tend1','ccheques_emitidos_rechazados_lag1','mcheques_emitidos_rechazados_delta2','cpayroll2_trx_tend1','ccheques_depositados_lag1','Master_cadelantosefectivo_delta1','tcuentas_delta1','ccajeros_propios_descuentos_delta2','ccheques_depositados_rechazados_tend1','ctarjeta_visa_descuentos_lag2','ccuenta_corriente_tend1','mcajeros_propios_descuentos_delta1','cpagodeservicios_tend1','mcheques_depositados_rechazados_lag1','Visa_madelantopesos_delta2','tcuentas_delta2','Visa_Finiciomora_delta2','cseguro_vivienda_delta1','mtarjeta_master_descuentos_tend1','active_quarter_delta1','Master_madelantodolares_delta1','minversion1_pesos_delta1','mprestamos_hipotecarios_delta2','mcajeros_propios_descuentos_lag2','Master_madelantodolares','cforex_buy_lag2','tpaquete1','ccajeros_propios_descuentos_lag2','tpaquete1_lag1','cforex_buy','Master_madelantopesos_delta2','mcheques_emitidos_rechazados_tend1','cplazo_fijo_delta2','Master_madelantodolares_delta2','mcajeros_propios_descuentos_delta2','mtarjeta_master_descuentos_delta2','Visa_madelantodolares_delta1','cseguro_auto_lag1','ctarjeta_master_lag1','ccheques_emitidos_rechazados_tend1','mpagodeservicios_lag2','mcajeros_propios_descuentos_tend1','ctarjeta_master_debitos_automaticos_delta1','cseguro_vivienda_delta2','Visa_cadelantosefectivo_tend1','mpagodeservicios_tend1','ccheques_depositados_delta2','Master_cadelantosefectivo','minversion1_pesos_delta2','ctarjeta_master_descuentos_lag2','cinversion1_lag1','cseguro_accidentes_personales_delta2','ctarjeta_visa_lag2','mcaja_ahorro_adicional_delta1','Visa_madelantodolares_lag1','minversion1_pesos_lag2','Master_madelantodolares_lag2','ccheques_emitidos_rechazados_lag2','Visa_madelantodolares','Visa_madelantodolares_lag2','Visa_cadelantosefectivo_delta2','cseguro_auto','mprestamos_prendarios_delta1','Visa_cadelantosefectivo_lag1','mprestamos_prendarios_lag2','Master_madelantodolares_lag1','ccajas_otras_lag1','Visa_cadelantosefectivo_delta1','Visa_madelantopesos_lag1','ctarjeta_master_lag2','Master_Finiciomora_delta2','tpaquete4_delta1','ctarjeta_debito_delta1','cpagodeservicios_lag2','ctarjeta_master_descuentos_delta2','mforex_buy_lag1','ccheques_emitidos_lag1','Master_cadelantosefectivo_lag2','cforex_buy_lag1','cinversion2_lag2','mcheques_depositados_delta1','Master_madelantopesos_delta1','mforex_buy_delta1','cplazo_fijo_delta1','Master_madelantopesos','active_quarter_delta2','mtarjeta_master_descuentos_delta1','minversion1_pesos_lag1','mcaja_ahorro_adicional_delta2','cliente_edad_delta1','cforex_delta1','mttarjeta_master_debitos_automaticos_delta1','Visa_madelantopesos_tend1','ctarjeta_master_debitos_automaticos_delta2','ctarjeta_visa_descuentos','minversion2_delta1','tcallcenter_lag1','ccajas_otras_lag2','mcaja_ahorro_adicional_lag1','mcaja_ahorro_adicional_lag2','Master_madelantopesos_lag2','ctarjeta_master_debitos_automaticos_lag2','Visa_madelantodolares_delta2','ccheques_emitidos_delta1','cinversion2_lag1','mcheques_depositados_lag1','Master_mconsumosdolares','tcuentas','cforex_buy_delta1','cseguro_vida','ccajas_depositos_lag2','Master_madelantopesos_lag1','mprestamos_prendarios_delta2','ccajeros_propios_descuentos_tend1','ccajas_depositos_delta2','cseguro_vida_lag2','Master_delinquency_lag2','ctarjeta_master','Master_mpagosdolares_tend1','tpaquete3_lag1','ctarjeta_master_debitos_automaticos','ccajas_depositos_delta1','cprestamos_hipotecarios_lag1','cforex_buy_delta2','cliente_antiguedad_delta1','ctarjeta_master_debitos_automaticos_lag1','catm_trx_other_delta1','cinversion1','ccheques_emitidos_lag2','mcaja_ahorro_adicional','cforex_sell_delta2','Master_msaldodolares_delta1','minversion2_lag1','cforex_sell_lag1','cforex_sell_delta1','mcajeros_propios_descuentos','ctarjeta_visa_descuentos_delta2','Master_status_lag2','ctarjeta_visa_descuentos_delta1','tpaquete3_delta1','Master_cadelantosefectivo_lag1','minversion1_pesos','Visa_madelantopesos_delta1','Visa_madelantopesos','ccaja_seguridad_delta2','cinversion2_tend1','mcheques_emitidos_delta1','cprestamos_hipotecarios','cseguro_auto_tend1','minversion2_lag2','cforex_lag1','ctarjeta_master_delta1','cforex','mforex_buy_lag2','cforex_sell','active_quarter_lag2','cseguro_vivienda_lag2','ccheques_emitidos_delta2','mprestamos_hipotecarios_lag2','ccajas_otras','cprestamos_prendarios_tend1','cplazo_fijo_lag1','cseguro_vida_lag1','minversion2_delta2','Master_msaldodolares_lag1','Visa_cadelantosefectivo','tcallcenter_lag2','ccheques_depositados_delta1','ccajas_depositos_lag1','Master_msaldodolares','ctarjeta_debito_delta2','Master_msaldodolares_tend1','mforex_buy_tend1','cseguro_vivienda_lag1','ccajas_consultas_lag1','ccajas_transacciones_lag2','Master_mpagado_lag1','ccajas_depositos','mprestamos_hipotecarios_tend1','tpaquete4_delta2','minversion1_pesos_tend1','Master_mpagosdolares','cprestamos_hipotecarios_lag2','Visa_cadelantosefectivo_lag2','Master_mpagosdolares_delta1','mforex_sell','mcheques_depositados_delta2','cseguro_vivienda','Master_mconsumosdolares_delta2','Master_delinquency_lag1','mttarjeta_master_debitos_automaticos_delta2','mcaja_ahorro_adicional_tend1','mcheques_emitidos_lag1','mv_mpagosdolares','mcheques_depositados_lag2','cprestamos_hipotecarios_tend1','cliente_edad_delta2','ccajas_extracciones_lag2','Master_mpagado','catm_trx_other_lag2','cforex_buy_tend1','ccuenta_debitos_automaticos_delta1','mcheques_emitidos_lag2','tcuentas_lag1','ccajas_extracciones','tpaquete3_lag2','ccajas_extracciones_delta2','Master_mpagosdolares_lag1','cplazo_fijo_lag2','mttarjeta_master_debitos_automaticos','mprestamos_prendarios_lag1','active_quarter','mvr_mpagosdolares','ccajas_extracciones_delta1','ccheques_depositados','Master_mconsumosdolares_tend1','ccheques_depositados_lag2','Master_msaldodolares_delta2','catm_trx_other_delta2','mforex_buy_delta2','cforex_delta2','cforex_sell_lag2','ctarjeta_visa_descuentos_tend1','Visa_status_tend1','mforex_buy','Master_delinquency_delta1','Visa_delinquency_delta1','tpaquete1_tend1','ccallcenter_transacciones_lag1','matm_other','cseguro_vivienda_tend1','Master_status_lag1','mforex_sell_delta1','mplazo_fijo_dolares_lag1','cforex_lag2','ccajas_transacciones_lag1','cinversion1_tend1','cprestamos_prendarios','cinversion2','mforex_sell_lag1','minversion2','ctarjeta_master_transacciones_delta2','Master_mpagado_lag2','Master_mconsumosdolares_delta1','Master_mconsumosdolares_lag2','Master_delinquency_delta2','mplazo_fijo_dolares_lag2','catm_trx_other','mforex_sell_lag2','mttarjeta_master_debitos_automaticos_lag1','mcheques_depositados','Master_mpagado_delta2','Master_msaldodolares_lag2','tcuentas_tend1','catm_trx_other_lag1','Master_mconsumosdolares_lag1','tpaquete4_lag2','ccuenta_debitos_automaticos_delta2','Master_status_tend1','Visa_delinquency_lag2','ccallcenter_transacciones_lag2','ctarjeta_master_transacciones_delta1','mcheques_emitidos','mv_status03','ccheques_emitidos','minversion2_tend1','ccajas_extracciones_lag1','Master_Finiciomora_lag2','cseguro_vida_tend1','mprestamos_hipotecarios','Visa_Finiciomora_lag2','Master_mpagosdolares_delta2','matm_other_delta1','Visa_mpagosdolares','tcuentas_lag2','Master_mpagado_tend1','Visa_status_lag2','Master_mpagado_delta1','Master_mpagosdolares_lag2','Master_status_delta1','mvr_msaldodolares','matm_other_delta2','mforex_sell_delta2','mprestamos_hipotecarios_lag1','cliente_vip_lag1','mttarjeta_master_debitos_automaticos_lag2','cseguro_accidentes_personales_lag1','ctarjeta_visa_delta2','matm_other_lag1','Visa_mpagosdolares_delta1','Visa_msaldodolares_delta1','mcheques_emitidos_delta2','mv_msaldodolares','ccajas_consultas_lag2','ccajas_transacciones','Master_Finiciomora_lag1','ctarjeta_master_transacciones_lag2','catm_trx_lag1','tpaquete4_lag1','cliente_antiguedad_delta2','Master_cconsumos_delta1','ctarjeta_visa_debitos_automaticos_delta1','Visa_mpagosdolares_lag1','cseguro_accidentes_personales','catm_trx_delta2','Visa_msaldodolares','Visa_msaldodolares_lag1','ccajas_otras_delta2','mtarjeta_master_consumo_delta1','mvr_msaldodolares2','Master_delinquency_tend1','thomebanking_lag1','cseguro_accidentes_personales_lag2','ctarjeta_visa_delta1','Master_cconsumos_delta2','Visa_msaldodolares_delta2','catm_trx_delta1','tpaquete3_delta2','ctarjeta_master_transacciones_lag1','Visa_mpagado','Visa_mpagosdolares_delta2','Master_status','matm_other_lag2','catm_trx_lag2','mcheques_emitidos_tend1','Master_cconsumos','Visa_status_delta2','ctarjeta_master_debitos_automaticos_tend1','cliente_vip_delta2','cextraccion_autoservicio_delta1','thomebanking_lag2','mv_mpagado','ccajas_transacciones_delta1','internet_lag2','ccuenta_debitos_automaticos_lag1','Visa_mpagosdolares_tend1','mtarjeta_master_consumo_delta2','ccuenta_debitos_automaticos_lag2','tcallcenter','Master_cconsumos_lag1','Master_mconsumototal_tend1','ctarjeta_master_descuentos_tend1','Visa_mpagado_lag1','ccajas_consultas','mttarjeta_master_debitos_automaticos_tend1','cextraccion_autoservicio_delta2','cliente_vip','Visa_mpagado_delta2','mvr_mpagado','thomebanking_delta1','mcheques_depositados_tend1','mprestamos_prendarios','mtarjeta_master_consumo_lag1','Visa_mpagosdolares_lag2','Visa_mpagado_delta1','Master_cconsumos_lag2','Master_cconsumos_tend1','Master_msaldototal_delta1','cprestamos_personales_lag1','mplazo_fijo_dolares_delta1','mtarjeta_master_consumo_lag2','Visa_delinquency_lag1','Master_mpagospesos_delta2','Visa_mpagado_lag2','ccaja_seguridad_lag2','ccajas_otras_delta1','ccajas_transacciones_delta2','Master_mpagospesos_tend1','cforex_tend1','Master_Fvencimiento_delta1','Master_fechaalta_delta1','ccajas_otras_tend1','matm_delta1','cliente_vip_delta1','cextraccion_autoservicio_lag2','mprestamos_prendarios_tend1','matm_delta2','Master_mpagospesos_delta1','Visa_madelantopesos_lag2','cforex_sell_tend1','thomebanking_delta2','mextraccion_autoservicio_delta1','Master_mconsumototal_delta2','ccomisiones_mantenimiento_lag2','Visa_mlimitecompra_delta1','mcuenta_debitos_automaticos_delta1','Master_mconsumototal_delta1','mforex_sell_tend1','cprestamos_personales_delta1','mv_mconsumosdolares','Master_delinquency','Visa_msaldodolares_tend1','ctarjeta_debito_lag1','ctarjeta_debito','matm_lag1','Master_Finiciomora','Visa_fechaalta_delta1','ctransferencias_emitidas_delta1','ctarjeta_master_transacciones','mvr_mconsumosdolares','Master_mconsumototal_lag1','matm_lag2','Visa_delinquency_delta2','ctransferencias_recibidas_delta1','ccomisiones_otras_delta1','cpayroll_trx_delta1','Master_mconsumototal','Visa_Finiciomora_delta1','Master_msaldopesos_delta2','ctarjeta_debito_tend1','cseguro_accidentes_personales_tend1','ctransferencias_emitidas_lag1','ccaja_ahorro_delta1','mtarjeta_master_consumo','mextraccion_autoservicio_delta2','Visa_delinquency_tend1','Master_msaldototal_delta2','Master_mlimitecompra_delta2','ccheques_emitidos_tend1','internet_lag1','ccheques_depositados_tend1','matm','ccajas_depositos_tend1','ctarjeta_visa_debitos_automaticos','Master_msaldopesos_delta1','cpayroll_trx_delta2','Master_mpagospesos_lag1','ctarjeta_visa_debitos_automaticos_delta2','Visa_mlimitecompra_delta2','tcallcenter_delta1','Master_mlimitecompra_delta1','Visa_mpagado_tend1','ccallcenter_transacciones','catm_trx','mcuenta_debitos_automaticos_delta2','ctarjeta_visa_debitos_automaticos_lag1','catm_trx_other_tend1','mplazo_fijo_dolares_delta2','Master_mpagospesos_lag2','Visa_mconsumosdolares_delta1','ctarjeta_debito_lag2','ctarjeta_debito_transacciones_delta1','internet_delta2','Master_mconsumototal_lag2','ctransferencias_emitidas_lag2','ccajas_consultas_delta2','ccomisiones_otras_delta2','cmobile_app_trx_delta1','Visa_msaldodolares_lag2','ccajas_consultas_delta1','tmobile_app_tend1','cliente_vip_lag2','tpaquete4_tend1','matm_other_tend1','Master_msaldototal_lag2','ctransferencias_recibidas_delta2','ccaja_ahorro_lag2','mv_status02','Visa_mconsumosdolares','tmobile_app_lag2','tpaquete3','mplazo_fijo_dolares_tend1','Visa_Fvencimiento_delta1','ctarjeta_visa_debitos_automaticos_lag2','ccajas_extracciones_tend1','cextraccion_autoservicio_lag1','ctransferencias_recibidas_lag1','Master_mpagospesos','mtransferencias_emitidas_lag2','ctransferencias_recibidas_lag2','Visa_status_delta1','ctransferencias_emitidas','Visa_delinquency','ccaja_ahorro_delta2','mcuenta_debitos_automaticos_lag2','Master_msaldopesos_lag2','ctarjeta_debito_transacciones_delta2','active_quarter_tend1','ctarjeta_master_transacciones_tend1','mttarjeta_visa_debitos_automaticos_delta1','Visa_mconsumosdolares_delta2','cprestamos_personales_lag2','ctarjeta_visa_transacciones_delta1','ctransferencias_emitidas_delta2','Visa_mconsumosdolares_lag2','ccomisiones_mantenimiento_lag1','ccaja_seguridad_lag1','Master_msaldopesos','mextraccion_autoservicio_lag2','ccomisiones_mantenimiento','tpaquete3_tend1','mtarjeta_master_consumo_tend1','ccajas_transacciones_tend1','ccaja_ahorro_lag1','catm_trx_tend1','matm_tend1','Master_msaldopesos_tend1','mttarjeta_visa_debitos_automaticos','Master_msaldopesos_lag1','ccajas_consultas_tend1','Master_msaldototal_tend1','ctransferencias_recibidas','cliente_edad_tend1','mcomisiones_mantenimiento_lag1','Master_mlimitecompra_tend1','mttarjeta_visa_debitos_automaticos_lag1','mtransferencias_emitidas_delta1','tpaquete4','mcomisiones_mantenimiento_delta1','mplazo_fijo_dolares','cmobile_app_trx_delta2','cplazo_fijo_tend1','Visa_mconsumosdolares_lag1','mcuenta_debitos_automaticos_lag1','mv_status01','Visa_cconsumos_lag1','Master_msaldototal_lag1','Visa_mlimitecompra_tend1','Master_msaldototal','mvr_Visa_mlimitecompra','Visa_cconsumos_delta1','tcallcenter_delta2','Master_Fvencimiento_delta2','cmobile_app_trx_lag1','Master_status_delta2','mtransferencias_emitidas_lag1','cmobile_app_trx_lag2','ctarjeta_visa','cplazo_fijo','ccallcenter_transacciones_delta1','mtarjeta_visa_consumo_lag2','Visa_cconsumos_delta2','mcomisiones_mantenimiento_delta2','tcallcenter_tend1','Visa_mconsumosdolares_tend1','ctarjeta_visa_transacciones_delta2','mtransferencias_emitidas_delta2','thomebanking','ccallcenter_transacciones_delta2','tmobile_app_lag1','mtransferencias_recibidas_delta1','mtransferencias_emitidas','mttarjeta_visa_debitos_automaticos_lag2','mtarjeta_visa_consumo_delta2','mtarjeta_visa_consumo_delta1','Visa_cconsumos','Visa_mconsumototal_delta2','Visa_mconsumototal_delta1','Visa_cconsumos_lag2','Master_fechaalta_delta2','Visa_Finiciomora_lag1','cmobile_app_trx','mdescubierto_preacordado_lag2','ctarjeta_debito_transacciones_lag2','internet_delta1','ccuenta_debitos_automaticos','cprestamos_personales_delta2','ccaja_seguridad','internet','ccuenta_debitos_automaticos_tend1','ccomisiones_mantenimiento_delta1','mextraccion_autoservicio','Visa_Fvencimiento_delta2','internet_tend1','mtransferencias_recibidas_lag2','ccallcenter_transacciones_tend1','mcomisiones_delta1','ccomisiones_mantenimiento_delta2','ccaja_seguridad_tend1','ccomisiones_otras_lag1','mttarjeta_visa_debitos_automaticos_delta2','cproductos_lag1','mv_status06','Visa_mconsumototal_lag2','cproductos_lag2','Visa_mconsumototal_tend1','mtransferencias_recibidas_delta2','Visa_mpagominimo_lag2','mvr_msaldopesos2','mcomisiones_otras_delta1','Visa_mpagominimo_delta1','mvr_mconsumospesos','Visa_mconsumototal')


#Me fije cuando aparecian con datadrifting+por encima de canarios

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


campos_lags  <- setdiff(  colnames(dataset) ,  c("clase_ternaria","clase01", "numero_de_cliente","foto_mes", campos_malos, campos_malos2) )

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


#creo vector con variables flash 

tipom <- c(mtarjeta_visa_consumo,mprestamos_personales,mv_msaldototal,Visa_mpagominimo,mtarjeta_visa_consumo_lag1,mv_status04,mactivos_margen,mv_msaldopesos,mcuenta_corriente,mvr_msaldopesos,mv_mpagospesos,mvr_msaldototal,Visa_msaldototal,mdescubierto_preacordado,mtransferencias_recibidas_lag1,mcuenta_corriente_lag1,mrentabilidad,mrentabilidad_annual_lag2,mrentabilidad_lag1,mactivos_margen_lag1,mrentabilidad_tend1,mv_mconsumototal,mrentabilidad_annual,mv_mpagominimo,mvr_mpagospesos,mprestamos_personales_lag2,mrentabilidad_lag2,mv_mconsumospesos,mcaja_ahorro_dolares,mactivos_margen_lag2,Master_mlimitecompra_lag2,mpayroll_tend1,Master_fechaalta_lag2,mcomisiones_otras,Visa_mlimitecompra,Visa_msaldototal_delta1,mcomisiones_lag2,Visa_msaldototal_lag1,Visa_mconsumototal_lag1,mpayroll,mpayroll_lag1,mtransferencias_recibidas)

tipoc <- c(ctarjeta_visa_transacciones,cpayroll_trx,cpayroll_trx_lag2,ccomisiones_otras,cpayroll_trx_lag1,ctarjeta_debito_transacciones,ctarjeta_debito_transacciones_lag1,cproductos_delta1,cextraccion_autoservicio,ctarjeta_visa_transacciones_lag1,ccaja_ahorro,ctarjeta_visa_transacciones_lag2,cproductos_delta2,cproductos,ccomisiones_otras_lag2,cprestamos_personales)

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
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01", campos_malos, campos_malos2) )


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


