#source("~/buckets/b1/R/822_epic.r")

#Necesita para correr en Google Cloud
#300 GB de memoria RAM
#300 GB de espacio en el disco local
#8 vCPU

#clase_binaria2   1={BAJA+2,BAJA+1}    0={CONTINUA}
#Entrena en a union de OCHO  meses de [202001, 202009] - { 202006 }  haciendo subsampling al 10% de los continua
#Testea en  { 202011 }
#Utiliza lag de orden 1 y delta lag
#estima automaticamente la cantidad de registros a enviar al medio de la meseta (en lugar de la prob de corte)

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


kexperimento  <- 1010   #NA si se corre la primera vez, un valor concreto si es para continuar procesando

kscript         <- "822_epic"

karch_dataset    <- "./datasets/dataset_epic_v007-cocientes.csv.gz"   #este dataset se genero en el script 811_dataset_epic.r

kapply_mes       <- c(202101)  #El mes donde debo aplicar el modelo

ktest_mes_hasta  <- 202011  #Esto es lo que uso para testing
ktest_mes_desde  <- 202011

ktrain_subsampling  <- 0.1   #el undersampling que voy a hacer de los continua

ktrain_mes_hasta    <- 202010  #Obviamente, solo puedo entrenar hasta 202011
ktrain_mes_desde    <- 201901  
ktrain_meses_malos  <- c( 202006 )  #meses que quiero excluir del entrenamiento


kgen_mes_hasta    <- 202011   #La generacion final para Kaggle, sin undersampling
kgen_mes_desde    <- 201901


kBO_iter    <-  150   #cantidad de iteraciones de la Optimizacion Bayesiana

#Aqui se cargan los hiperparametros
hs <- makeParamSet( 
  makeNumericParam("learning_rate",    lower=    0.02 , upper=    0.08),#CHECK
  makeNumericParam("feature_fraction", lower=    0.2  , upper=    0.6), #CHECK 
  makeIntegerParam("min_data_in_leaf", lower= 800L   , upper= 5000L), #CHECK
  makeIntegerParam("num_leaves",       lower=  500L   , upper= 1400L) #CHECK
)

campos_malos  <- c("foto_mes","numero_de_cliente","mpasivos_margen","mcuentas_saldo","mautoservicio", "cpagomiscuentas", "mpagomiscuentas", "mtarjeta_visa_descuentos", "ctrx_quarter", "Master_mfinanciacion_limite", "Master_mconsumospesos", "Master_fultimo_cierre","Master_mpagominimo","Visa_mfinanciacion_limite","Visa_msaldopesos","Visa_mconsumospesos", "Visa_fultimo_cierre", "Visa_mpagospesos","Visa_mconsumototal_lag2","mprestamos_personales_delta2/mcaja_ahorro_lag1","mv_mpagominimo_lag2","mv_mfinanciacion_limite_lag2","mvr_msaldopesos_lag2","mv_msaldototal_lag2","mvr_mpagospesos/ctrx_quarter_lag1","Visa_fultimo_cierre_lag2","mprestamos_personales_lag2","mv_mfinanciacion_limite_lag1","mtransferencias_emitidas_lag2","ctarjeta_visa_transacciones_lag2","mcaja_ahorro_dolares_lag1","mvr_msaldototal_lag2","mcuenta_debitos_automaticos_lag2","mv_mpagospesos_lag1","tmobile_app_lag2","mtarjeta_visa_consumo/mcaja_ahorro_lag1","mvr_Master_mlimitecompra_lag1","ctarjeta_visa_debitos_automaticos_lag2","mprestamos_personales/mcaja_ahorro_lag1","Visa_mconsumototal_lag1","mautoservicio_lag2","Master_mlimitecompra_lag1","mvr_msaldototal_lag1","mcuenta_debitos_automaticos_lag1","ctarjeta_visa_debitos_automaticos_lag1","mv_mpagospesos/mcaja_ahorro_lag1","ctarjeta_visa_transacciones/mcaja_ahorro_lag1","mv_mpagospesos_lag2","mttarjeta_visa_debitos_automaticos_lag1","mpagomiscuentas_lag2","mvr_msaldopesos_lag1","Visa_mconsumosdolares_lag2","cproductos_lag2","mtarjeta_visa_consumo_lag2","tmobile_app_lag1","mextraccion_autoservicio_lag2","ctrx_quarter_lag1/mtarjeta_visa_consumo","Visa_msaldototal_lag2","mvr_Visa_mlimitecompra_lag2","mcaja_ahorro_lag1/mprestamos_personales","mvr_mconsumospesos_lag2","mtransferencias_emitidas_lag1","Visa_mconsumospesos_lag1","mcaja_ahorro_lag1/mcuenta_corriente","Visa_msaldopesos_lag1","mvr_Visa_mlimitecompra_lag1","ctarjeta_debito_transacciones_lag1","mcaja_ahorro_lag1/mprestamos_personales_delta2","Visa_msaldototal_lag1","mvr_mconsumototal_lag1","mtarjeta_visa_consumo_lag1","mdescubierto_preacordado_lag1","Master_msaldopesos_lag2","Visa_cconsumos_lag1","mprestamos_personales_delta2/cpayroll_trx_lag1","mcaja_ahorro_lag1/ctarjeta_visa_transacciones","mv_cconsumos_lag2","Master_msaldototal_lag1","mv_msaldopesos_lag2","Master_msaldototal_lag2","mvr_mconsumospesos_lag1","mvr_mconsumototal_lag2","ccaja_ahorro_lag1","Master_mpagominimo_lag1","mprestamos_personales/ctrx_quarter_lag1","mvr_mconsumosdolares_lag2","matm_lag2","Visa_mconsumosdolares_lag1","mv_msaldototal_lag1","mcaja_ahorro_lag1/ctarjeta_debito_transacciones","mcaja_ahorro_lag1/mtarjeta_visa_consumo","mv_mconsumospesos_lag2","ctarjeta_debito_transacciones/ctrx_quarter_lag1","mdescubierto_preacordado_lag2","cpagomiscuentas_lag1","cpayroll_trx_lag1/ctarjeta_debito_transacciones","Master_msaldopesos_lag1","mcuenta_corriente/cpayroll_trx_lag1","ctrx_quarter_lag1/ctarjeta_debito_transacciones","cpayroll_trx_lag1/mprestamos_personales_delta2","Visa_delinquency_lag2","Master_mconsumospesos_lag2","mv_mconsumototal_lag2","internet_lag1","ctarjeta_debito_transacciones_lag2","ccaja_ahorro_lag2","cpayroll_trx_lag1/mpayroll","cpayroll_trx_lag1/ctarjeta_visa_transacciones","ctransferencias_recibidas_lag2","mextraccion_autoservicio_lag1","mcaja_ahorro_lag1/cpayroll_trx_lag1","Master_mpagominimo_lag2","cextraccion_autoservicio_lag2","Master_mpagospesos_lag1","thomebanking_lag1","cliente_vip_lag1","mvr_msaldodolares2_lag2","mcheques_emitidos_rechazados_lag1","matm_other_lag2","mcaja_ahorro/cpayroll_trx_lag1","matm_lag1","mv_cconsumos_lag1","Master_mpagospesos_lag2","Master_mconsumototal_lag2","internet_lag2","ctrx_quarter_lag1/mprestamos_personales_delta2","mpayroll/cpayroll_trx_lag1","mv_mconsumospesos_lag1","mpayroll/ctrx_quarter_lag1","ctarjeta_debito_lag2","cprestamos_personales_lag1","mv_mconsumototal_lag1","matm_other_lag1","mvr_mconsumosdolares_lag1","ctarjeta_debito_transacciones/cpayroll_trx_lag1","ctransferencias_recibidas_lag1","ctarjeta_visa_transacciones/cpayroll_trx_lag1","ccomisiones_mantenimiento_lag2","ctransferencias_emitidas_lag2","Master_mconsumospesos_lag1","Master_Finiciomora_isNA_lag1","mvr_mpagospesos/cpayroll_trx_lag1","cpayroll_trx_lag1/mcaja_ahorro","cprestamos_personales_lag2","cpayroll_trx_lag1/mcaja_ahorro_lag1","cextraccion_autoservicio_lag1","mplazo_fijo_dolares_lag1","ctarjeta_master_transacciones_lag1","mprestamos_prendarios_lag1","mpayroll_lag1","mv_mconsumosdolares_lag1","cpayroll_trx_lag1/mv_mpagospesos","ctrx_quarter_lag1/mpayroll","mv_mconsumosdolares_lag2","thomebanking_lag2","mdescubierto_preacordado_delta2/cpayroll_trx_lag1","ccheques_emitidos_lag1","Master_cconsumos_lag2","ccajas_extracciones_lag1","cpayroll_trx_lag1/mprestamos_personales","cpayroll_trx/mcaja_ahorro_lag1","mprestamos_prendarios_lag2","mcaja_ahorro_lag1/mdescubierto_preacordado_delta2","cpayroll_trx_lag1/mcuenta_corriente","Master_mconsumototal_lag1","ctarjeta_debito_lag1","mprestamos_personales/cpayroll_trx_lag1","mpayroll/mcaja_ahorro_lag1","Master_cconsumos_lag1","ctransferencias_emitidas_lag1","mv_status01/cpayroll_trx_lag1","mttarjeta_master_debitos_automaticos_lag2","mplazo_fijo_dolares_lag2","ccallcenter_transacciones_lag1","mv_Finiciomora_lag2","ctrx_quarter_lag1/cpayroll_trx_lag1","ccuenta_debitos_automaticos_lag2","ccomisiones_mantenimiento_lag1","mtarjeta_master_consumo_lag2","mvr_msaldodolares2_lag1","mtarjeta_visa_consumo/cpayroll_trx_lag1","tpaquete4_lag2","cpagomiscuentas_lag2","mdescubierto_preacordado_delta2/ctrx_quarter_lag1","cpayroll_trx_lag1/mtarjeta_visa_consumo","mv_status01/ctrx_quarter_lag1","catm_trx_other_lag2","cseguro_accidentes_personales_lag2","mtarjeta_master_consumo_lag1","cpayroll_trx_lag2","catm_trx_lag1","ccajas_consultas_lag2","cpayroll_trx_lag1/ctrx_quarter_lag1","mcaja_ahorro_lag1/mpayroll","mprestamos_hipotecarios_lag1","tpaquete4_lag1","catm_trx_lag2","ctarjeta_master_transacciones_lag2","cpayroll_trx/cpayroll_trx_lag1","ccallcenter_transacciones_lag2","mttarjeta_master_debitos_automaticos_lag1","mcheques_emitidos_rechazados_lag2","cseguro_vida_lag1","cpayroll_trx_lag1/mvr_mpagospesos","mprestamos_hipotecarios_lag2","Master_Finiciomora_lag1","mvr_mpagado_lag2","mcaja_ahorro_lag1/mv_status01","mforex_sell_lag2","ccuenta_debitos_automaticos_lag1","mforex_sell_lag1","mvr_msaldodolares_lag1","ccheques_emitidos_rechazados_lag2","cseguro_vida_lag2","mcheques_depositados_lag2","mcheques_emitidos_lag1","catm_trx_other_lag1","ccheques_emitidos_rechazados_lag1","ccajas_depositos_lag2","ccajas_otras_lag1","mcaja_ahorro_lag1/cpayroll_trx","ccajas_otras_lag2","mvr_mpagosdolares_lag2","mpagodeservicios_lag2","mv_mpagospesos/cpayroll_trx_lag1","Visa_mpagosdolares_lag2","ccajas_extracciones_lag2","ctrx_quarter_lag1/mdescubierto_preacordado_delta2","ctrx_quarter_lag1/cpayroll_trx","mvr_mpagosdolares_lag1","ccajas_transacciones_lag1","mvr_msaldodolares_lag2","mv_msaldodolares_lag2","mvr_mpagado_lag1","Visa_mpagosdolares_lag1","Visa_msaldodolares_lag2","cseguro_accidentes_personales_lag1","ccheques_emitidos_lag2","Visa_mpagado_lag1","Visa_msaldodolares_lag1","mv_mpagado_lag1","Master_mconsumosdolares_lag2","Master_delinquency_lag1","ccajas_transacciones_lag2","Master_mpagosdolares_lag2","mv_mpagosdolares_lag2","ccajas_consultas_lag1","cforex_lag2","cforex_lag1","mcheques_depositados_rechazados_lag1","mv_mpagosdolares_lag1","mforex_buy_lag1","ccheques_depositados_lag2","tcallcenter_lag2","cpayroll_trx/ctrx_quarter_lag1","tcallcenter_lag1","mpagodeservicios_lag1","mtarjeta_visa_descuentos_lag2","mcheques_emitidos_lag2","ctarjeta_master_debitos_automaticos_lag2","mcheques_depositados_lag1","Master_mconsumosdolares_lag1","mv_status01/mcaja_ahorro_lag1","mv_status01_lag1","ccajas_depositos_lag1","Visa_mpagado_lag2","Visa_delinquency_lag1","ctarjeta_master_debitos_automaticos_lag1","ctarjeta_visa_descuentos_lag1","mv_msaldodolares_lag1","cpayroll_trx_lag1/cpayroll_trx","cpayroll_trx_lag1/mv_status01","mv_mpagado_lag2","Master_mconsumospesos_isNA_lag1","Master_mpagado_lag2","ccheques_depositados_lag1","mtarjeta_visa_descuentos_lag1","cprestamos_hipotecarios_lag1","Visa_Finiciomora_lag2","mv_status04_lag1","Master_msaldodolares_lag2","ctarjeta_visa_lag2","mv_status01_lag2","ctarjeta_visa_lag1","cseguro_vivienda_lag1","cplazo_fijo_lag2","mcajeros_propios_descuentos_lag1","Master_mpagosdolares_lag1","cseguro_auto_lag2","cpayroll_trx_lag1","cseguro_vivienda_lag2","mforex_buy_lag2","mcaja_ahorro_adicional_lag2","cforex_sell_lag2","cpayroll_trx_lag1/mdescubierto_preacordado_delta2","mtarjeta_visa_descuentos_isNA_lag1","cprestamos_prendarios_lag2","cpagodeservicios_lag1","minversion2_lag2","cplazo_fijo_lag1","Visa_madelantodolares_lag2","minversion1_pesos_lag1","mv_status07_lag1","tcuentas_lag2","ccheques_depositados_rechazados_lag2","cprestamos_hipotecarios_lag2","active_quarter_lag2","Visa_Finiciomora_isNA_lag1","mtarjeta_master_descuentos_lag1","minversion2_lag1","Master_msaldodolares_lag1","ctarjeta_master_descuentos_lag2","Master_Finiciomora_lag2","minversion1_pesos_lag2","mcaja_ahorro_adicional_lag1","Master_mpagado_lag1","ctarjeta_visa_descuentos_lag2","tpaquete3_lag1","ccajeros_propios_descuentos_lag1","ccheques_depositados_rechazados_lag1","Master_madelantopesos_lag2","cpagodeservicios_lag2","tpaquete3_lag2","Visa_status_lag1","mv_status05_lag2","mv_status03_lag2","mv_status05_lag1","Visa_mconsumospesos_isNA_lag1","mtarjeta_visa_descuentos_isNA_lag2","ctarjeta_master_descuentos_lag1","mcheques_depositados_rechazados_lag2","cinversion2_lag2","mv_status06_lag2","cinversion1_lag1","Master_madelantodolares_isNA_lag2","mv_status04_lag2","mv_status02_lag1","mtarjeta_master_descuentos_lag2","Visa_madelantopesos_lag1","Master_cadelantosefectivo_lag2","Master_status_lag1","ccajeros_propios_descuentos_lag2","Master_delinquency_lag2","cinversion2_lag1","cforex_sell_lag1","Master_madelantopesos_lag1","Master_madelantodolares_isNA_lag1","Master_mpagosdolares_isNA_lag2","mvr_madelantopesos_lag1","Visa_cadelantosefectivo_lag1","mcajeros_propios_descuentos_lag2","Visa_status_lag2","active_quarter_lag1","Visa_mfinanciacion_limite_isNA_lag2","Visa_fultimo_cierre_isNA_lag2","Master_cadelantosefectivo_lag1","cinversion1_lag2","cpayroll2_trx_lag1","cseguro_auto_lag1","cprestamos_prendarios_lag1","mv_madelantopesos_lag2","mv_status06_lag1","Master_Fvencimiento_isNA_lag2","Master_madelantopesos_isNA_lag1","mpayroll2_lag1","cforex_buy_lag2","cforex_buy_lag1","Visa_cadelantosefectivo_lag2","Visa_mconsumospesos_isNA_lag2","Master_madelantopesos_isNA_lag2","Visa_madelantodolares_lag1","mv_status02_lag2","Visa_madelantopesos_lag2","Visa_Finiciomora_isNA_lag2","Master_mconsumospesos_isNA_lag2","Master_delinquency_isNA_lag2","tpaquete3_tend","Visa_mconsumosdolares_tend","mttarjeta_visa_debitos_automaticos_delta1","ccallcenter_transacciones_delta1","mtransferencias_recibidas_delta2","mtransferencias_recibidas_tend","Visa_mpagominimo_delta2","Visa_cconsumos_delta1","mv_mconsumospesos_tend","mprestamos_prendarios","ccajas_transacciones_tend","mv_fultimo_cierre_tend","mpagomiscuentas_delta1","ccuenta_debitos_automaticos","Visa_mpagospesos_delta1","mvr_msaldopesos_delta2","Master_mconsumospesos_tend","Master_mlimitecompra_tend","mv_msaldopesos_delta2","Visa_mconsumospesos_delta2","Visa_msaldopesos_delta1","mvr_mpagospesos_delta1","Visa_mconsumototal_delta1","mvr_Master_mlimitecompra_tend","Visa_mconsumototal_delta2","cextraccion_autoservicio_tend","Visa_delinquency_delta2","ctarjeta_visa_debitos_automaticos_delta2","mtarjeta_master_consumo_tend","mtransferencias_emitidas_delta1","mvr_msaldopesos_delta1","ctarjeta_visa_debitos_automaticos","ctransferencias_recibidas_tend","mv_mfinanciacion_limite_tend","mcomisiones_mantenimiento_delta1","mv_mconsumototal","mv_msaldototal_tend","Master_cconsumos_tend","Master_mpagominimo_tend","tmobile_app_delta1","internet_delta1","mcuenta_debitos_automaticos_delta2","ccallcenter_transacciones","mv_mconsumototal_delta2","internet_tend","mautoservicio_tend","mvr_mconsumototal_delta2","mvr_msaldototal_delta2","mvr_mconsumospesos_delta1","tcallcenter","mv_mconsumototal_tend","internet_delta2","ccajas_otras_tend","cprestamos_personales_delta1","mv_msaldototal_delta2","mplazo_fijo_dolares_tend","mv_status06_tend","mautoservicio_delta1","ctarjeta_debito_transacciones_delta2","mv_status06_delta2","mvr_msaldototal_delta1","mdescubierto_preacordado_delta1","mvr_msaldodolares2","catm_trx_other_tend","ccajas_extracciones_tend","ctarjeta_debito_tend","ctransferencias_recibidas","ctarjeta_master_transacciones","ctransferencias_emitidas_delta2","Visa_msaldototal_delta1","mtransferencias_recibidas_delta1","matm_tend","mtransferencias_emitidas_delta2","Visa_mconsumosdolares","mextraccion_autoservicio_tend","ccajas_consultas_delta2","mextraccion_autoservicio_delta2","ccomisiones_otras_delta2","tpaquete3_delta1","Visa_mlimitecompra_tend","Master_mpagospesos_tend","mplazo_fijo_dolares_delta2","cpagomiscuentas_delta2","Master_delinquency_tend","mv_Fvencimiento_delta2","ccajas_depositos_tend","Master_msaldototal_tend","ctarjeta_debito_transacciones_delta1","Visa_status_delta2","Master_msaldopesos_tend","Visa_fechaalta_delta2","Visa_msaldodolares","Master_mconsumototal_tend","Master_Finiciomora_delta1","mv_mlimitecompra_tend","mv_mpagominimo_delta1","ctarjeta_debito","mv_status05_delta2","mextraccion_autoservicio_delta1","Master_mconsumototal","cliente_vip_delta1","mv_cconsumos_delta1","ccomisiones_otras_delta1","Visa_mconsumosdolares_delta1","matm_other_tend","mvr_mconsumototal_delta1","mv_mconsumospesos_delta2","cliente_vip_delta2","mprestamos_hipotecarios","Visa_mconsumosdolares_delta2","matm","matm_delta2","catm_trx_tend","mplazo_fijo_dolares_delta1","Master_mfinanciacion_limite_delta2","Master_mlimitecompra_delta2","Visa_fultimo_cierre_delta1","mv_status07_lag2","cextraccion_autoservicio_delta2","Master_fultimo_cierre_delta2","mv_msaldopesos_delta1","mv_mconsumosdolares_tend","mv_mconsumosdolares_delta2","mv_mpagospesos_delta1","matm_other_delta2","mvr_mconsumosdolares","Visa_mfinanciacion_limite_delta1","Master_fultimo_cierre_delta1","tcallcenter_delta2","Master_mpagominimo_delta2","cliente_edad_tend","Visa_fultimo_cierre_delta2","mvr_msaldodolares2_delta2","ctransferencias_emitidas","ctarjeta_master_transacciones_tend","Master_mconsumosdolares_tend","ccajas_consultas","Master_mpagominimo_delta1","cforex_sell_tend","Visa_mpagado_tend","Master_status_delta2","cforex_tend","Master_mfinanciacion_limite_delta1","ccaja_seguridad_lag2","mv_mfinanciacion_limite_delta1","mv_msaldototal_delta1","mvr_msaldodolares2_delta1","mttarjeta_master_debitos_automaticos_tend","Master_msaldopesos_delta2","cplazo_fijo_tend","Master_msaldototal_delta2","ctarjeta_master_transacciones_delta2","mvr_mconsumosdolares_delta2","mv_fultimo_cierre_delta2","ctransferencias_emitidas_delta1","mv_mfinanciacion_limite_delta2","minversion1_pesos","mv_mconsumospesos_delta1","Visa_mpagado_delta2","ccajas_consultas_delta1","Visa_Fvencimiento_delta2","mv_fultimo_cierre_delta1","ctransferencias_recibidas_delta2","Master_msaldopesos_delta1","mv_msaldodolares_tend","mvr_mconsumosdolares_delta1","Visa_mlimitecompra_delta2","matm_delta1","matm_other_delta1","cprestamos_hipotecarios","mv_status07_delta1","ctarjeta_master_debitos_automaticos_tend","Master_mpagospesos_delta1","mcheques_emitidos_rechazados","mv_fechaalta_delta2","Master_mconsumospesos_delta2","Visa_mfinanciacion_limite_delta2","mv_mlimitecompra_delta2","Master_cconsumos","mv_status05_delta1","cextraccion_autoservicio_delta1","mv_mconsumototal_delta1","ctarjeta_debito_delta2","mcheques_emitidos","ccajas_depositos","ccheques_emitidos_rechazados_tend","ccheques_emitidos","Visa_msaldodolares_tend","Master_mconsumospesos_delta1","mv_status04_tend","Visa_mpagosdolares_tend","Master_mpagospesos_delta2","mforex_sell","Master_status_tend","mpagodeservicios_tend","Master_msaldototal_delta1","thomebanking_delta2","mv_status02_tend","cpagodeservicios_tend","mcheques_emitidos_tend","mcheques_depositados_tend","mv_mpagosdolares_tend","mcuenta_debitos_automaticos_delta1","ctarjeta_master_lag1","Master_mlimitecompra_delta1","mv_mpagado_tend","Master_Fvencimiento_delta2","ccuenta_debitos_automaticos_delta2","catm_trx_delta2","ctarjeta_master_transacciones_delta1","mforex_sell_tend","Visa_madelantodolares_tend","ctarjeta_visa_debitos_automaticos_delta1","cpayroll_trx_delta1","tcallcenter_delta1","Visa_mlimitecompra_delta1","Master_delinquency_delta2","Master_mconsumototal_delta1","ccheques_emitidos_tend","cforex_buy_tend","ccheques_depositados_tend","mtarjeta_visa_descuentos_tend","mv_status04_delta1","catm_trx_other_delta2","mforex_sell_delta2","mvr_msaldodolares_delta2","mforex_buy_tend","mtarjeta_master_consumo_delta1","mttarjeta_master_debitos_automaticos","mtarjeta_master_consumo_delta2","mcheques_emitidos_rechazados_tend","ctarjeta_visa_descuentos_tend","Master_mpagosdolares_tend","mcheques_emitidos_delta2","matm_other","cseguro_accidentes_personales","mvr_mpagado","Master_cconsumos_delta2","mvr_mpagado_delta2","mttarjeta_master_debitos_automaticos_delta2","Master_delinquency","mv_mconsumosdolares","active_quarter_tend","Visa_msaldodolares_delta2","ccajas_otras","mv_mconsumosdolares_delta1","Master_msaldodolares_tend","Master_cconsumos_delta1","mvr_Visa_mlimitecompra_delta2","ctarjeta_master_descuentos_tend","mvr_mpagado_delta1","Master_Finiciomora_tend","Master_mconsumosdolares","ccheques_emitidos_delta2","ccajas_transacciones_delta1","mvr_Master_mlimitecompra_delta2","mforex_buy_delta2","catm_trx_delta1","mcheques_emitidos_rechazados_delta2","mv_mpagado","cinversion1","Master_fechaalta_delta2","catm_trx_other_delta1","Visa_mpagado_delta1","mcheques_depositados_rechazados_tend","cprestamos_prendarios_tend","mtarjeta_visa_descuentos_delta1","ctransferencias_recibidas_delta1","mforex_buy","mv_mpagado_delta2","mv_Fvencimiento_delta1","mv_status01_delta2","Visa_fechaalta_delta1","thomebanking_delta1","mprestamos_prendarios_tend","mv_mlimitecompra_delta1","minversion2","Master_Fvencimiento_delta1","Master_mconsumototal_delta2","active_quarter","mcheques_depositados","tpaquete4_tend","ccajas_otras_delta1","mforex_sell_delta1","Visa_mpagosdolares_delta2","Visa_cadelantosefectivo_tend","ccajas_otras_delta2","ccajas_extracciones","mvr_mpagosdolares_delta2","Visa_delinquency_delta1","cseguro_vida","minversion2_tend","ccajas_extracciones_delta2","ctarjeta_visa_descuentos_delta1","cinversion2","ccheques_emitidos_rechazados","mv_msaldodolares_delta2","mv_msaldodolares","mcheques_depositados_delta2","ccajas_depositos_delta2","minversion1_pesos_tend","catm_trx_other","mv_fechaalta_delta1","mv_status03_tend","cpagomiscuentas_delta1","ctarjeta_master_tend","mv_msaldodolares_delta1","mvr_mpagosdolares","mcaja_ahorro_adicional","mvr_msaldodolares_delta1","ccaja_seguridad_tend","ctarjeta_visa_descuentos","Visa_mpagosdolares_delta1","ccheques_depositados_rechazados_tend","mvr_msaldodolares","ccajas_transacciones_delta2","Visa_msaldodolares_delta1","Visa_Fvencimiento_delta1","mcajeros_propios_descuentos_delta1","mv_status02","mv_status06_delta1","mv_status03_delta2","cplazo_fijo_delta2","Visa_mpagado","Master_mpagado_tend","mprestamos_prendarios_delta2","ccajas_transacciones","Master_fechaalta_delta1","mv_mpagosdolares_delta2","minversion2_delta2","mv_status02_delta2","mvr_mpagosdolares_delta1","mv_status02_delta1","ccajas_extracciones_delta1","mcheques_depositados_delta1","cforex_delta2","cforex","mcajeros_propios_descuentos","mttarjeta_master_debitos_automaticos_delta1","ccheques_emitidos_rechazados_delta2","ccheques_depositados","tcuentas_tend","mcheques_emitidos_rechazados_delta1","Visa_mpagosdolares","cprestamos_prendarios","cliente_vip","mtarjeta_master_descuentos_delta1","mv_mpagado_delta1","Master_mpagosdolares_delta1","ctarjeta_debito_delta1","mvr_Master_mlimitecompra_delta1","mcajeros_propios_descuentos_tend","Master_mpagosdolares","ccajas_depositos_delta1","ctarjeta_master_descuentos_delta2","cforex_buy_delta2","mcheques_emitidos_delta1","mforex_buy_delta1","Master_mpagado_delta2","mtarjeta_visa_descuentos_delta2","Master_mconsumosdolares_delta2","mvr_Visa_mlimitecompra_delta1","Master_mpagado_delta1","cseguro_vivienda_tend","Master_mpagosdolares_delta2","cpayroll2_trx_tend","cseguro_accidentes_personales_tend","mpagodeservicios_delta2","Master_madelantopesos_tend","tcuentas_lag1","ccajeros_propios_descuentos_delta1","mprestamos_hipotecarios_delta2","ctarjeta_master_descuentos_delta1","Master_mpagado","ctarjeta_master_debitos_automaticos","cforex_delta1","Master_cadelantosefectivo_tend","Master_delinquency_delta1","ccajeros_propios_descuentos_tend","mcajeros_propios_descuentos_delta2","mv_Finiciomora_delta2","Master_msaldodolares","Master_mconsumosdolares_delta1","ctarjeta_master_debitos_automaticos_delta2","ccaja_seguridad_delta1","mprestamos_hipotecarios_tend","Visa_Finiciomora_delta2","cliente_edad_delta2","Master_msaldodolares_delta2","cforex_sell_delta2","mv_mpagosdolares","Master_msaldodolares_delta1","minversion2_delta1","ccheques_depositados_delta1","ccheques_depositados_delta2","Visa_madelantopesos_tend","Master_madelantopesos","mcaja_ahorro_adicional_tend","mtarjeta_master_descuentos_tend","ccajeros_propios_descuentos_delta2","ccuenta_debitos_automaticos_delta1","cinversion2_delta2","cinversion1_tend","cforex_buy_delta1","mv_mpagosdolares_delta1","Master_cadelantosefectivo","mv_status05","mpagodeservicios_delta1","mpayroll2_tend","ctarjeta_visa_descuentos_delta2","cforex_sell","mpagodeservicios","cinversion2_tend","cseguro_accidentes_personales_delta2","mcaja_ahorro_adicional_delta2","mcheques_depositados_rechazados_delta2","mv_status03_delta1","mv_status03","Visa_status_delta1","ctarjeta_master","mv_cadelantosefectivo_tend","cforex_sell_delta1","ccaja_seguridad_delta2","minversion1_pesos_delta2","mcaja_ahorro_adicional_delta1","cplazo_fijo_delta1","tcuentas","cpagodeservicios","Master_status","ccheques_emitidos_rechazados_delta1","Master_madelantodolares_lag2","Master_madelantodolares_lag1","ccheques_depositados_rechazados_delta1","mvr_madelantodolares_lag1","cliente_edad_delta1","mprestamos_prendarios_delta1","cseguro_vivienda","ctarjeta_master_delta2","active_quarter_delta2","mcheques_depositados_rechazados_delta1","ccajeros_propios_descuentos","ccheques_emitidos_delta1","ctarjeta_master_debitos_automaticos_delta1","mv_madelantopesos_tend","cprestamos_prendarios_delta2","Master_madelantodolares_tend","mtarjeta_master_descuentos","Visa_madelantopesos","cinversion1_delta1","cseguro_auto","Master_cadelantosefectivo_delta2","mtarjeta_master_descuentos_delta2","Master_madelantodolares","mcheques_depositados_rechazados","cseguro_vivienda_delta2","cseguro_vida_tend","cseguro_vida_delta2","Visa_madelantopesos_delta1","mvr_madelantopesos_lag2","Visa_madelantopesos_delta2","Visa_cadelantosefectivo","ccheques_depositados_rechazados","Master_cadelantosefectivo_delta1","cinversion1_delta2","cinversion2_delta1","cprestamos_prendarios_delta1","cforex_buy","Master_status_delta1","active_quarter_delta1","ctarjeta_master_lag2","cpagodeservicios_delta2","minversion1_dolares","mv_cadelantosefectivo_delta2","ctarjeta_visa_delta1","mvr_madelantodolares_lag2","Master_madelantodolares_delta2","ctarjeta_master_descuentos","mv_madelantopesos_delta2","minversion1_dolares_lag1","tcuentas_delta2","Master_status_lag2","mvr_madelantopesos_delta1","mv_cadelantosefectivo_delta1","Master_madelantopesos_delta1","mv_status03_lag1","cpagodeservicios_delta1","mvr_madelantopesos_delta2","Master_Finiciomora_delta2","minversion1_pesos_delta1","minversion1_dolares_tend","mvr_madelantopesos","tpaquete1_tend","cseguro_vivienda_delta1","cseguro_auto_tend","mplazo_fijo_pesos","ccheques_depositados_rechazados_delta2","Visa_cadelantosefectivo_delta2","ctarjeta_master_delta1","tpaquete4_delta1","mplazo_fijo_pesos_lag2","cprestamos_hipotecarios_tend","minversion1_dolares_lag2","mv_cadelantosefectivo_lag2","Master_madelantopesos_delta2","tpaquete1_delta2","Visa_madelantodolares")   #aqui se deben cargar todos los campos culpables del Data Drifting

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

particionar  <- function( data,  division, agrupa="",  campo="fold", start=1, seed=NA )
{
  if( !is.na(seed) )   set.seed( seed )
  
  bloque  <- unlist( mapply(  function(x,y) { rep( y, x )} ,   division,  seq( from=start, length.out=length(division) )  ) )  
  
  data[ ,  (campo) :=  sample( rep( bloque, ceiling(.N/length(bloque))) )[1:.N],
        by= agrupa ]
}
#------------------------------------------------------------------------------

HemiModelos  <- function( hparam )
{
  
  #Construyo el modelo sobre el fold=1
  dgeneracion1  <- lgb.Dataset( data=    data.matrix(  dataset[ generacion_final==1 & fold==1, campos_buenos, with=FALSE]),
                                label=   dataset[ generacion_final==1 & fold==1, clase01] )
  
  modelo_final1  <- lightgbm( data= dgeneracion1,
                              param= hparam,
                              verbose= -100 )
  
  rm( dgeneracion1 )  #borro y libero memoria
  gc()
  
  #Aplico el modelo al fold=2
  prediccion  <- predict( modelo_final1, data.matrix( dataset[ generacion_final==1 & fold==2, campos_buenos, with=FALSE]) )
  dataset[ generacion_final==1 & fold==2, prob := prediccion ]
  
  tb_modelitos[ dataset[ generacion_final==1 & fold==2], 
                on=c("numero_de_cliente","foto_mes"),  
                paste0( "E", kexperimento,"_", GLOBAL_iteracion ) := i.prob  ]
  
  
  #AHORA SOBRE LA OTRA MITAD -----------------
  
  #Construyo el modelo sobre el fold=2
  dgeneracion2  <- lgb.Dataset( data=    data.matrix(  dataset[ generacion_final==1 & fold==2, campos_buenos, with=FALSE]),
                                label=   dataset[ generacion_final==1 & fold==2, clase01]
  )
  
  modelo_final2  <- lightgbm( data= dgeneracion2,
                              param= hparam,
                              verbose= -100
  )
  
  rm( dgeneracion2 )  #borro y libero memoria
  gc()
  
  #Aplico el modelo al fold=1
  prediccion  <- predict( modelo_final2, data.matrix( dataset[ generacion_final==1 & fold==1, campos_buenos, with=FALSE]) )
  dataset[ generacion_final==1 & fold==1, prob := prediccion ]
  
  tb_modelitos[ dataset[ generacion_final==1 & fold==1], 
                on= c("numero_de_cliente","foto_mes"),  
                paste0( "E", kexperimento,"_", GLOBAL_iteracion ) := i.prob  ]
  
  dataset[  , prob := NULL ]
  
}
#------------------------------------------------------------------------------

FullModelo  <- function( hparam )
{
  #entreno sin undersampling
  dgeneracion  <- lgb.Dataset( data=    data.matrix(  dataset[ generacion_final==1 , campos_buenos, with=FALSE]),
                               label=   dataset[ generacion_final==1, clase01]
  )
  
  modelo_final  <- lightgbm( data= dgeneracion,
                             param= hparam,
                             verbose= -100
  )
  
  rm( dgeneracion )  #borro y libero memoria
  gc()
  
  #calculo la importancia de variables
  tb_importancia  <- lgb.importance( model= modelo_final )
  fwrite( tb_importancia, 
          file= paste0( kimp, "imp_", sprintf("%03d", GLOBAL_iteracion), ".txt"),
          sep="\t" )
  
  #Aplico sobre todo el dataset
  prediccion  <- predict( modelo_final, data.matrix( dataset[  , campos_buenos, with=FALSE]) )
  dataset[ , prob := prediccion ]
  tb_modelitos[ dataset, 
                on=c("numero_de_cliente","foto_mes"),  
                paste0( "E", kexperimento,"_", GLOBAL_iteracion ) := i.prob  ]
  
  #Fin primera pasada modelitos
  
  prediccion  <- predict( modelo_final, data.matrix( dapply[  , campos_buenos, with=FALSE]) )
  
  predsort  <- sort(prediccion, decreasing=TRUE)
  pos_corte  <- as.integer(hparam$ratio_corte*nrow(dapply))
  prob_corte <- predsort[ pos_corte ]
  Predicted  <- as.integer( prediccion > prob_corte )
  
  entrega  <- as.data.table( list( "numero_de_cliente"= dapply$numero_de_cliente, 
                                   "Predicted"= Predicted)  )
  
  #genero el archivo para Kaggle
  fwrite( entrega, 
          file= paste0(kkaggle, sprintf("%03d", GLOBAL_iteracion), ".csv" ),
          sep= "," )
  
  base  <- round( pos_corte / 500 ) * 500   - 3000
  evaluados  <- c( seq(from=base, to=pmax(base+6000,15000), by=500 ) , pos_corte )  
  evaluados  <- sort( evaluados )
  
  for(  pos  in  evaluados )
  {
    prob_corte  <-  predsort[ pos ]
    Predicted  <- as.integer( prediccion > prob_corte )
    
    entrega  <- as.data.table( list( "numero_de_cliente"= dapply$numero_de_cliente, 
                                     "Predicted"= Predicted)  )
    
    #genero el archivo para Kaggle
    fwrite( entrega, 
            file= paste0(kkagglemeseta, sprintf("%03d", GLOBAL_iteracion), 
                         "_",  sprintf( "%05d", pos) ,".csv" ),
            sep= "," )
  }
  
  rm( entrega, Predicted )
}
#------------------------------------------------------------------------------

VPOS_CORTE  <- c()

fganancia_lgbm_meseta  <- function(probs, datos) 
{
  vlabels  <- getinfo(datos, "label")
  vpesos   <- getinfo(datos, "weight")
  
  #solo sumo 48750 si vpesos > 1, hackeo 
  tbl  <- as.data.table( list( "prob"=probs, "gan"= ifelse( vlabels==1 & vpesos > 1, 48750, -1250 ) ) )
  
  setorder( tbl, -prob )
  tbl[ , posicion := .I ]
  tbl[ , gan_acum :=  cumsum( gan ) ]
  setorder( tbl, -gan_acum )   #voy por la meseta
  
  gan  <- mean( tbl[ 1:10,  gan_acum] )  #meseta de tama?o 10
  
  pos_meseta  <- tbl[ 1:10,  median(posicion)]
  VPOS_CORTE  <<- c( VPOS_CORTE, pos_meseta )
  
  return( list( "name"= "ganancia", 
                "value"=  gan,
                "higher_better"= TRUE ) )
}
#------------------------------------------------------------------------------

x  <- list( "learning_rate"= 0.02, 
            "feature_fraction"= 0.50,
            "min_data_in_leaf"= 4000,
            "num_leaves"= 600 )


EstimarGanancia_lightgbm  <- function( x )
{
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1
  
  gc()
  
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
                          num_iterations= 9999,   #un numero muy grande, lo limita early_stopping_rounds
                          force_row_wise= TRUE    #para que los alumnos no se atemoricen con tantos warning
  )
  
  #el parametro discolo, que depende de otro
  param_variable  <- list(  early_stopping_rounds= as.integer(50 + 1/x$learning_rate) )
  
  param_completo  <- c( param_basicos, param_variable, x )
  
  VPOS_CORTE  <<- c()
  set.seed( 999983 )
  modelo  <- lgb.train( data= dtrain,
                        valids= list( valid= dvalid ),
                        eval= fganancia_lgbm_meseta,
                        param= param_completo,
                        verbose= -100 )
  
  #unlist(modelo$record_evals$valid$ganancia$eval)[ modelo$best_iter ]
  
  #calculo la ganancia sobre los datos de testing
  prediccion  <- predict( modelo, data.matrix( dataset[ test==1, campos_buenos, with=FALSE]) )
  
  tb_test  <- as.data.table( list( "ganancia"=dataset[ test==1, ifelse(clase_ternaria=="BAJA+2", 48750, -1250)],
                                   "prob"= prediccion ) )
  
  setorder( tb_test, -prob )
  ganancia  <- tb_test[  1:   VPOS_CORTE[ modelo$best_iter ], sum( ganancia ) ]
  
  attr(ganancia,"extras" )  <- list("num_iterations"= modelo$best_iter)  #esta es la forma de devolver un parametro extra
  
  param_final  <- copy( param_completo )
  param_final["early_stopping_rounds"]  <- NULL
  param_final$num_iterations <- modelo$best_iter  #asigno el mejor num_iterations
  param_final$ratio_corte  <- VPOS_CORTE[ modelo$best_iter ] / nrow( dvalid )
  
  
  #si tengo una ganancia superadora, genero el archivo para Kaggle
  if( ganancia > GLOBAL_ganancia_max)
  {
    GLOBAL_ganancia_max  <<- ganancia  #asigno la nueva maxima  a una variable GLOBAL, por eso el <<-
    
    FullModelo( param_final )
    HemiModelos( param_final )
    fwrite( tb_modelitos, file= kmodelitos, sep= "," )
  }
  
  #logueo 
  xx  <- param_final
  xx$iteracion_bayesiana  <- GLOBAL_iteracion
  xx$ganancia  <- ganancia  #le agrego la ganancia
  loguear( xx,  arch= klog )
  
  return( ganancia )
}
#------------------------------------------------------------------------------
#Aqui empieza el programa

if( is.na(kexperimento ) )   kexperimento <- get_experimento()  #creo el experimento

#en estos archivos quedan los resultados
dir.create( paste0( "./work/E",  kexperimento, "/" ) )     #creo carpeta del experimento dentro de work
dir.create( paste0( "./kaggle/E",  kexperimento, "/" ) )   #creo carpeta del experimento dentro de kaggle
dir.create( paste0( "./kaggle/E",  kexperimento, "/meseta/" ) )   #creo carpeta del experimento dentro de kaggle

kbayesiana    <- paste0("./work/E",  kexperimento, "/E",  kexperimento, "_", kscript, ".RDATA" )
klog          <- paste0("./work/E",  kexperimento, "/E",  kexperimento, "_", kscript, "_BOlog.txt" )
kimp          <- paste0("./work/E",  kexperimento, "/E",  kexperimento, "_", kscript, "_" )
kkaggle       <- paste0("./kaggle/E",kexperimento, "/E",  kexperimento, "_", kscript, "_" )
kkagglemeseta <- paste0("./kaggle/E",kexperimento, "/meseta/E",  kexperimento, "_", kscript, "_" )
kmodelitos    <- paste0("./modelitos/E", kexperimento, "_modelitos.csv.gz" )



#cargo el dataset que tiene los 36 meses
dataset  <- fread(karch_dataset)

#si ya existe el archivo log, traigo hasta donde llegue
if( file.exists(klog) )
{
  tabla_log  <- fread( klog)
  GLOBAL_iteracion  <- nrow( tabla_log ) -1
  GLOBAL_ganancia_max  <- tabla_log[ , max(ganancia) ]
  
  tb_modelitos  <- fread( kmodelitos )
} else {
  GLOBAL_iteracion  <- 0
  GLOBAL_ganancia_max  <- -Inf
  
  tb_modelitos  <- dataset[  ,  c("numero_de_cliente","foto_mes"), with=FALSE ]
  fwrite( tb_modelitos, file= kmodelitos, sep= "," )
}


#creo vector con variables flash 

#tipom <- c("mcaja_ahorro","mdescubierto_preacordado_delta2","mtarjeta_visa_consumo","cpayroll_trx","mv_mpagospesos","mv_status01","ctrx_quarter_lag1","ctarjeta_visa_transacciones","mcaja_ahorro_lag1","ctarjeta_debito_transacciones","mpayroll","mvr_mpagospesos","mprestamos_personales_delta2","mcuenta_corriente","mprestamos_personales","cpayroll_trx_lag1")


#agrego un quinto de canaritos
for( i  in 1:(ncol(dataset)/5))  dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset))]


#cargo los datos donde voy a aplicar el modelo
dapply  <- copy( dataset[  foto_mes %in% kapply_mes ] )


#creo la clase_binaria2   1={ BAJA+2,BAJA+1}  0={CONTINUA}
dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]


particionar( dataset, division= c(1,1), agrupa= c("foto_mes","clase_ternaria" ), seed= ksemilla_azar )

dataset[    foto_mes>= kgen_mes_desde  &
              foto_mes<= kgen_mes_hasta & 
              !( foto_mes %in% ktrain_meses_malos ),
            generacion_final:= 1L ]  #donde entreno


#Defino los datos donde entreno, con subsampling de los CONTINUA
vector_azar  <- runif( nrow(dataset) )
dataset[    foto_mes>= ktrain_mes_desde  &
              foto_mes<= ktrain_mes_hasta & 
              !( foto_mes %in% ktrain_meses_malos ) & 
              ( clase01==1 | vector_azar < ktrain_subsampling ),  
            entrenamiento:= 1L ]  #donde entreno


#defino donde valido
dataset[    foto_mes>= ktest_mes_desde &
              foto_mes<= ktest_mes_hasta &
              fold== 1,
            validacion:= 1L ]  #donde entreno

#defino donde testeo
dataset[    foto_mes>= ktest_mes_desde &
              foto_mes<= ktest_mes_hasta &
              fold== 2,
            test:= 1L ]  #donde entreno



#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), 
                           c("clase_ternaria","clase01", "generacion_final", "entrenamiento", "validacion", "test", "fold", campos_malos) )

#dejo los datos en el formato que necesita LightGBM
#uso el weight como un truco ESPANTOSO para saber la clase real
dtrain  <- lgb.Dataset( data=    data.matrix(  dataset[ entrenamiento==1 , campos_buenos, with=FALSE]),
                        label=   dataset[ entrenamiento==1, clase01],
                        weight=  dataset[ entrenamiento==1, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)] ,
                        free_raw_data= TRUE
)

dvalid  <- lgb.Dataset( data=    data.matrix(  dataset[validacion==1 , campos_buenos, with=FALSE]),
                        label=   dataset[ validacion==1, clase01],
                        weight=  dataset[ validacion==1, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)] ,
                        free_raw_data= TRUE
)


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


