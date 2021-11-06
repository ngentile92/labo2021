#Necesita para correr en Google Cloud
#256 GB de memoria RAM
#300 GB de espacio en el disco local
#8 vCPU


#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("Rcpp")
require("rlist")
require("yaml")

require("lightgbm")


#defino la carpeta donde trabajo
directory.root  <-  "~/buckets/b1/"  #Google Cloud
setwd( directory.root )

palancas  <- list()  #variable con las palancas para activar/desactivar

palancas$version  <- "v007-cocientes"   #Muy importante, ir cambiando la version

palancas$variablesdrift  <- c("mttarjeta_visa_debitos_automaticos_lag1","mpagomiscuentas_lag2","mvr_msaldopesos_lag1","mv_mpagospesos_delta1","Visa_mconsumosdolares_lag2","cproductos_lag2","mtarjeta_visa_consumo_lag2","Visa_mconsumototal_delta2","ctarjeta_visa_transacciones/mcuenta_corriente","cpayroll_trx/mpayroll","ctarjeta_debito_transacciones/mtarjeta_visa_consumo","tmobile_app_lag1","mtransferencias_recibidas_delta2","Visa_cconsumos_delta1","internet_delta1","mextraccion_autoservicio_lag2","ctrx_quarter_lag1/mtarjeta_visa_consumo","mtarjeta_visa_consumo/mcuenta_corriente","tpaquete4","ctarjeta_debito_transacciones/mdescubierto_preacordado_delta2","Visa_msaldototal_lag2","mvr_Visa_mlimitecompra_lag2","mcaja_ahorro_lag1/mprestamos_personales","mpayroll","mdescubierto_preacordado_delta2/ctarjeta_visa_transacciones","mvr_mconsumospesos_lag2","mtransferencias_emitidas_lag1","ctarjeta_visa_transacciones/mcaja_ahorro","ccomisiones_otras_delta2","Visa_mconsumospesos_lag1","mcaja_ahorro_lag1/mcuenta_corriente","ccuenta_debitos_automaticos","Master_mpagospesos","mvr_mconsumospesos","Visa_msaldopesos_lag1","mvr_msaldototal_delta1","mv_msaldopesos_delta1","mvr_Visa_mlimitecompra_lag1","ctarjeta_debito_transacciones_lag1","mplazo_fijo_dolares_delta2","cpagomiscuentas_delta2","mprestamos_personales_delta2/mtarjeta_visa_consumo","mtarjeta_visa_consumo/mprestamos_personales","mprestamos_personales_delta2/mpayroll","Visa_fechaalta_delta2","mv_mconsumospesos","mcaja_ahorro_lag1/mprestamos_personales_delta2","mdescubierto_preacordado_delta2/mtarjeta_visa_consumo","mv_fultimo_cierre_delta1","mv_mpagospesos/mtarjeta_visa_consumo","Visa_Fvencimiento_delta2","Visa_msaldototal_lag1","mprestamos_prendarios","mvr_mconsumospesos_delta1","mv_mpagospesos/mcuenta_corriente","Visa_mconsumospesos_delta1","ccomisiones_otras_delta1","mv_mpagospesos/mprestamos_personales_delta2","mvr_mconsumototal_lag1","mtarjeta_visa_consumo_lag1","mdescubierto_preacordado_lag1","mautoservicio_delta2","Master_msaldopesos_lag2","Visa_msaldototal_delta1","ctarjeta_debito_transacciones/mv_mpagospesos","cpayroll_trx/mprestamos_personales_delta2","Visa_cconsumos_lag1","ctarjeta_visa_debitos_automaticos","mprestamos_personales_delta2/cpayroll_trx_lag1","ctarjeta_debito_transacciones_delta2","mprestamos_personales_delta2/ctarjeta_visa_transacciones","mcaja_ahorro_lag1/ctarjeta_visa_transacciones","Visa_cconsumos","ctransferencias_emitidas_delta2","mv_cconsumos_lag2","Visa_mconsumosdolares_delta2","cprestamos_personales_delta1","Master_msaldototal_lag1","mv_status06_delta1","mv_msaldopesos_lag2","Master_msaldototal_lag2","mv_status01/mprestamos_personales","Master_fultimo_cierre_delta1","mdescubierto_preacordado_delta2/mvr_mpagospesos","mvr_mconsumospesos_lag1","mvr_mconsumototal_lag2","ccaja_ahorro_lag1","mvr_msaldodolares2","Master_mpagominimo_lag1","mprestamos_personales/mcuenta_corriente","mvr_mconsumototal_delta1","ctransferencias_recibidas","mprestamos_personales/ctrx_quarter_lag1","ctarjeta_debito_transacciones/mvr_mpagospesos","mautoservicio_delta1","ctarjeta_visa_transacciones/mprestamos_personales","mvr_mconsumosdolares_lag2","matm_lag2","cpayroll_trx/mcuenta_corriente","Visa_mconsumosdolares_lag1","mv_msaldototal_lag1","mcaja_ahorro_lag1/ctarjeta_debito_transacciones","Master_msaldototal_delta2","mcaja_ahorro_lag1/mtarjeta_visa_consumo","mv_mconsumospesos_lag2","ccajas_consultas_delta2","ctarjeta_debito_transacciones/ctrx_quarter_lag1","tcallcenter_delta2","Visa_mconsumosdolares","mdescubierto_preacordado_lag2","ccajas_consultas_delta1","Master_mlimitecompra_delta2","mtransferencias_recibidas_delta1","Master_msaldopesos_delta2","cpagomiscuentas_lag1","cpayroll_trx_lag1/ctarjeta_debito_transacciones","Visa_mfinanciacion_limite_delta2","Master_mpagominimo_delta2","mv_status07_delta1","mv_cconsumos_delta1","mtarjeta_master_consumo_delta2","Master_msaldopesos_lag1","mcuenta_corriente/cpayroll_trx_lag1","mextraccion_autoservicio_delta2","Master_Finiciomora_isNA","mv_status01","ccuenta_debitos_automaticos_delta2","mv_status01/mcuenta_corriente","mextraccion_autoservicio_delta1","ctrx_quarter_lag1/ctarjeta_debito_transacciones","mv_mfinanciacion_limite_delta1","cpayroll_trx_lag1/mprestamos_personales_delta2","Visa_delinquency_lag2","Master_mconsumospesos_lag2","mv_mconsumototal_lag2","ccajas_consultas","mvr_mconsumototal","internet_lag1","ctarjeta_debito_transacciones_lag2","Master_Finiciomora_delta1","Master_mconsumototal","mprestamos_personales/mdescubierto_preacordado_delta2","ccaja_ahorro_lag2","Visa_mlimitecompra_delta2","cpayroll_trx_lag1/mpayroll","minversion1_pesos","Master_mfinanciacion_limite_delta2","cpayroll_trx_lag1/ctarjeta_visa_transacciones","ctransferencias_recibidas_lag2","mextraccion_autoservicio_lag1","Master_Finiciomora_isNA_delta2","cpayroll_trx/mdescubierto_preacordado_delta2","Master_mpagospesos_delta2","mcaja_ahorro_lag1/cpayroll_trx_lag1","mvr_mconsumosdolares","mv_msaldototal_delta1","Master_mpagominimo_lag2","cextraccion_autoservicio_lag2","Master_fultimo_cierre_delta2","Master_mpagospesos_lag1","mtarjeta_visa_consumo/mprestamos_personales_delta2","thomebanking_lag1","matm_delta2","ctarjeta_visa_debitos_automaticos_delta1","ctarjeta_debito_transacciones_delta1","cliente_vip_lag1","Visa_fultimo_cierre_delta1","mvr_msaldodolares2_lag2","Visa_delinquency_delta1","mcheques_emitidos_rechazados_lag1","matm_other_lag2","matm","ctarjeta_visa_transacciones/ctarjeta_debito_transacciones","Master_mpagominimo_delta1","Master_mlimitecompra_delta1","ctarjeta_debito_transacciones/ctarjeta_visa_transacciones","mcaja_ahorro/cpayroll_trx_lag1","ctarjeta_master_transacciones_delta2","matm_lag1","mvr_msaldodolares2_delta1","Visa_mconsumototal_delta1","mprestamos_personales/ctarjeta_visa_transacciones","mv_cconsumos_lag1","mpayroll/mcuenta_corriente","Master_mpagospesos_lag2","Master_mconsumototal_lag2","mv_mpagospesos/ctarjeta_debito_transacciones","internet_lag2","ctrx_quarter_lag1/mprestamos_personales_delta2","mpayroll/cpayroll_trx_lag1","mplazo_fijo_dolares_delta1","Master_msaldototal_delta1","mv_mconsumospesos_lag1","mpayroll/ctrx_quarter_lag1","ctarjeta_debito_lag2","cextraccion_autoservicio_delta2","ctarjeta_debito_delta2","mcaja_ahorro/mpayroll","cprestamos_personales_lag1","Visa_mconsumosdolares_delta1","mv_mconsumototal_lag1","mdescubierto_preacordado_delta2/mv_status01","ctarjeta_master_delta1","ctarjeta_debito_transacciones","mv_status04_delta1","mprestamos_hipotecarios","mprestamos_personales_delta2/mvr_mpagospesos","mv_mconsumototal_delta1","matm_other_lag1","Visa_mlimitecompra_delta1","mvr_mconsumosdolares_lag1","ctarjeta_debito_transacciones/cpayroll_trx_lag1","matm_other_delta2","ctransferencias_recibidas_lag1","ctarjeta_visa_transacciones/cpayroll_trx_lag1","Master_fechaalta_delta2","ccomisiones_mantenimiento_lag2","ctransferencias_emitidas_lag2","mdescubierto_preacordado_delta2/mv_mpagospesos","mvr_mpagospesos/mprestamos_personales","Master_mconsumospesos_lag1","ctarjeta_visa_transacciones/mpayroll","Master_Finiciomora_isNA_lag1","Visa_mfinanciacion_limite_delta1","mpayroll/mvr_mpagospesos","Visa_fultimo_cierre_delta2","mvr_mpagospesos/cpayroll_trx_lag1","ctransferencias_emitidas_delta1","cpayroll_trx_lag1/mcaja_ahorro","mvr_mpagospesos/mpayroll","ctransferencias_emitidas","mtarjeta_visa_consumo/ctarjeta_debito_transacciones","cprestamos_personales_lag2","cpayroll_trx_lag1/mcaja_ahorro_lag1","mv_status01/mprestamos_personales_delta2","cextraccion_autoservicio_lag1","mplazo_fijo_dolares_lag1","mv_mconsumospesos_delta1","mvr_mconsumosdolares_delta1","ctarjeta_master_transacciones_lag1","Master_mfinanciacion_limite_delta1","mprestamos_prendarios_lag1","thomebanking_delta1","ctransferencias_recibidas_delta1","ctarjeta_debito","catm_trx","matm_delta1","ctransferencias_recibidas_delta2","mpayroll_lag1","mv_mconsumosdolares_lag1","mvr_mpagospesos/mprestamos_personales_delta2","mv_mconsumosdolares","cpayroll_trx_lag1/mv_mpagospesos","ctrx_quarter_lag1/mpayroll","mforex_sell","mcheques_emitidos","mv_mconsumosdolares_lag2","Master_mconsumospesos_delta2","thomebanking_lag2","mdescubierto_preacordado_delta2/cpayroll_trx_lag1","ccheques_emitidos_lag1","Master_cconsumos_lag2","ccajas_extracciones_lag1","cpayroll_trx_lag1/mprestamos_personales","cpayroll_trx/mcaja_ahorro_lag1","mprestamos_prendarios_lag2","mcaja_ahorro_lag1/mdescubierto_preacordado_delta2","Master_Fvencimiento_delta2","ctarjeta_visa_transacciones/mprestamos_personales_delta2","cpayroll_trx_lag1/mcuenta_corriente","Master_mconsumototal_lag1","Visa_mpagado","ctarjeta_master_transacciones_delta1","ccajas_depositos","ctarjeta_debito_lag1","Master_mconsumospesos_delta1","mprestamos_personales/cpayroll_trx_lag1","mv_status01/ctarjeta_debito_transacciones","Visa_mpagado_delta2","Visa_Finiciomora_isNA_delta1","mpayroll/mcaja_ahorro_lag1","cextraccion_autoservicio","Master_cconsumos_lag1","Master_msaldopesos_delta1","mpayroll/mv_mpagospesos","Master_status","ctransferencias_emitidas_lag1","mv_mconsumosdolares_delta1","mv_status01/cpayroll_trx_lag1","mttarjeta_master_debitos_automaticos_lag2","mprestamos_personales/mv_mpagospesos","mplazo_fijo_dolares_lag2","ccallcenter_transacciones_lag1","matm_other_delta1","mv_Finiciomora_lag2","Master_delinquency","ccajas_transacciones_delta2","mv_status02","ctrx_quarter_lag1/cpayroll_trx_lag1","Master_mpagospesos_delta1","mv_status05_delta1","mprestamos_personales/mtarjeta_visa_consumo","ccuenta_debitos_automaticos_lag2","ccomisiones_mantenimiento_lag1","mtarjeta_master_consumo_lag2","mvr_msaldodolares2_lag1","mvr_mpagado","mtarjeta_visa_consumo/cpayroll_trx_lag1","mdescubierto_preacordado_delta2/mcuenta_corriente","mttarjeta_master_debitos_automaticos_delta2","cliente_vip_delta1","mcuenta_corriente/mv_status01","cliente_vip_delta2","mpayroll/mcaja_ahorro","Visa_delinquency_delta2","tpaquete4_lag2","cpagomiscuentas_lag2","mprestamos_personales_delta2/cpayroll_trx","ccheques_emitidos","ccajas_extracciones","cprestamos_personales","mforex_sell_delta2","mdescubierto_preacordado_delta2/ctrx_quarter_lag1","cpayroll_trx_lag1/mtarjeta_visa_consumo","cpayroll_trx","mv_status01/ctrx_quarter_lag1","tcallcenter_delta1","mcaja_ahorro/cpayroll_trx","catm_trx_delta2","thomebanking_delta2","Master_cconsumos","mprestamos_personales/mvr_mpagospesos","mprestamos_personales/mv_status01","catm_trx_other_lag2","mvr_msaldodolares_delta1","Visa_fechaalta_delta1","cseguro_accidentes_personales_lag2","mtarjeta_master_consumo_lag1","ctarjeta_visa_descuentos","mprestamos_personales_delta2/mv_status01","cpayroll_trx_lag2","Master_mconsumosdolares","catm_trx_lag1","ccajas_consultas_lag2","cpayroll_trx_lag1/ctrx_quarter_lag1","mcaja_ahorro_lag1/mpayroll","cextraccion_autoservicio_delta1","mcuenta_debitos_automaticos_delta1","mforex_buy","mprestamos_hipotecarios_lag1","tpaquete4_lag1","mvr_mpagado_delta1","catm_trx_lag2","ctarjeta_master_transacciones_lag2","cpayroll_trx/mcaja_ahorro","mcheques_emitidos_rechazados","catm_trx_other_delta2","cpayroll_trx/cpayroll_trx_lag1","ccallcenter_transacciones_lag2","mttarjeta_master_debitos_automaticos_lag1","ccaja_seguridad_delta2","mcheques_emitidos_rechazados_lag2","mcheques_depositados","cseguro_vida_lag1","cpayroll_trx_lag1/mvr_mpagospesos","Master_mpagosdolares","mprestamos_hipotecarios_lag2","catm_trx_delta1","Master_fechaalta_delta1","minversion2","mtarjeta_master_consumo_delta1","mcheques_emitidos_delta2","cpagomiscuentas_delta1","mtarjeta_visa_descuentos_delta1","mforex_sell_delta1","Master_Finiciomora_lag1","mv_mpagospesos/mv_status01","Master_cconsumos_delta2","Visa_msaldototal_isNA_delta2","ccajas_otras_delta1","mv_fechaalta_delta1","ccajas_otras","cprestamos_hipotecarios","mvr_mpagado_lag2","mcaja_ahorro_lag1/mv_status01","mforex_sell_lag2","ccuenta_debitos_automaticos_lag1","Visa_msaldodolares_delta2","mforex_sell_lag1","mvr_msaldodolares_lag1","Visa_mpagosdolares_delta2","cinversion1","ccheques_emitidos_rechazados_lag2","cseguro_vida_lag2","ctarjeta_debito_delta1","mvr_mpagospesos/cpayroll_trx","mcheques_depositados_lag2","cinversion2","mcheques_emitidos_lag1","catm_trx_other_lag1","mv_mpagado_delta1","ccheques_emitidos_rechazados_lag1","ccheques_emitidos_rechazados","mvr_msaldodolares","ccajas_transacciones","mv_mlimitecompra_delta1","ccajas_depositos_lag2","mv_mpagado","ccajas_otras_lag1","mcaja_ahorro_lag1/cpayroll_trx","ccajas_otras_lag2","mvr_mpagosdolares_lag2","mpagodeservicios_lag2","Master_mconsumototal_delta2","mv_mpagospesos/mdescubierto_preacordado_delta2","mv_mpagospesos/cpayroll_trx_lag1","Visa_mpagosdolares_lag2","ccajas_extracciones_delta1","catm_trx_other_delta1","ccajas_extracciones_lag2","ccheques_emitidos_delta2","ctrx_quarter_lag1/mdescubierto_preacordado_delta2","ctrx_quarter_lag1/cpayroll_trx","mvr_mpagosdolares_lag1","Visa_mpagado_delta1","cplazo_fijo_delta2","matm_other","ccajas_transacciones_lag1","Visa_Fvencimiento_delta1","mvr_msaldodolares_lag2","cseguro_accidentes_personales","mv_msaldodolares","mv_status01/mvr_mpagospesos","ccajas_transacciones_delta1","ctarjeta_master_debitos_automaticos_delta2","mv_msaldodolares_lag2","mvr_mpagado_lag1","Visa_msaldodolares","Visa_mpagosdolares_lag1","Visa_msaldodolares_lag2","cseguro_accidentes_personales_lag1","ccheques_emitidos_lag2","Visa_mpagado_lag1","ccajas_extracciones_delta2","Visa_msaldodolares_lag1","ctarjeta_visa_descuentos_delta1","mcheques_emitidos_delta1","mcheques_emitidos_rechazados_delta2","mv_mpagado_lag1","mvr_mpagosdolares","Master_mconsumosdolares_lag2","ccheques_depositados","Master_mconsumototal_delta1","Master_delinquency_lag1","ccajas_transacciones_lag2","Master_delinquency_delta1","mttarjeta_master_debitos_automaticos_delta1","Master_mpagosdolares_lag2","Visa_msaldodolares_delta1","mv_mpagosdolares_lag2","ccajas_consultas_lag1","cforex_lag2","Master_Fvencimiento_delta1","cforex_lag1","mv_Fvencimiento_delta1","mcajeros_propios_descuentos_delta1","mv_mpagospesos/mpayroll","mvr_Master_mlimitecompra_delta1","ctarjeta_master","mcheques_depositados_rechazados_lag1","mv_mpagosdolares_lag1","mforex_buy_lag1","mcheques_depositados_delta2","cliente_edad_delta2","ccheques_depositados_lag2","tcallcenter_lag2","mprestamos_prendarios_delta2","cpayroll_trx/ctrx_quarter_lag1","tcallcenter_lag1","mv_msaldodolares_delta1","mforex_buy_delta1","mpayroll/mprestamos_personales_delta2","ccheques_emitidos_rechazados_delta2","minversion2_delta2","mforex_buy_delta2","cprestamos_prendarios","mvr_mpagosdolares_delta1","Visa_delinquency_isNA_delta2","mpayroll/ctarjeta_visa_transacciones","ccajas_otras_delta2","ctarjeta_debito_transacciones/mv_status01","mpagodeservicios_lag1","cforex","cforex_buy_delta1","mtarjeta_visa_descuentos_lag2","mcheques_emitidos_lag2","ctarjeta_master_debitos_automaticos_lag2","mttarjeta_master_debitos_automaticos","mv_mpagospesos/cpayroll_trx","mcheques_depositados_lag1","Master_mconsumosdolares_lag1","mv_status01/mcaja_ahorro_lag1","mv_status01_lag1","Visa_mpagosdolares","ccajas_depositos_lag1","Visa_mpagado_lag2","Visa_delinquency_lag1","ctarjeta_master_debitos_automaticos_lag1","Visa_status_delta2","ctarjeta_visa","ctarjeta_visa_descuentos_lag1","ctarjeta_visa_transacciones/cpayroll_trx","catm_trx_other","Visa_mpagosdolares_delta1","mv_msaldodolares_lag1","ctarjeta_visa_transacciones/mdescubierto_preacordado_delta2","mtarjeta_visa_descuentos_delta2","mcajeros_propios_descuentos_delta2","ccajeros_propios_descuentos_delta1","ccajas_depositos_delta1","cpayroll_trx_lag1/cpayroll_trx","ccuenta_debitos_automaticos_delta1","Master_status_delta1","cpayroll_trx_lag1/mv_status01","cseguro_vivienda_delta2","Master_mfinanciacion_limite_isNA_delta2","active_quarter","Master_mconsumospesos_isNA_delta2","mvr_Visa_mlimitecompra_delta1","mv_mpagado_lag2","Master_madelantopesos","Master_mconsumospesos_isNA_lag1","Master_mpagado_lag2","Master_msaldodolares","ccajas_depositos_delta2","mcheques_depositados_delta1","mcaja_ahorro_adicional","ctarjeta_master_debitos_automaticos","ccheques_depositados_lag1","Master_cconsumos_delta1","Master_delinquency_delta2","cforex_delta1","cseguro_vida","ctarjeta_visa_descuentos_delta2","mtarjeta_visa_descuentos_lag1","cprestamos_hipotecarios_lag1","mtarjeta_master_descuentos_delta1","Visa_Finiciomora_lag2","mv_status04_lag1","cpayroll_trx_delta1","Master_msaldodolares_lag2","ctarjeta_visa_lag2","Master_status_delta2","cinversion2_delta2","mv_status01_lag2","mv_status05","ctarjeta_visa_lag1","cseguro_vivienda_lag1","Master_status_isNA_delta2","cplazo_fijo_lag2","cforex_sell","Master_mconsumosdolares_delta1","mcajeros_propios_descuentos_lag1","mcajeros_propios_descuentos","cforex_buy_delta2","Master_mpagosdolares_lag1","cseguro_auto_lag2","mv_status03_delta1","cpayroll_trx_lag1","cseguro_vivienda_lag2","Master_mpagado","mforex_buy_lag2","Master_mconsumosdolares_delta2","mv_mpagosdolares","cprestamos_prendarios_delta2","mpagodeservicios_delta2","mcaja_ahorro_adicional_lag2","cforex_sell_lag2","Master_mpagado_delta2","mv_status03","cpayroll_trx_lag1/mdescubierto_preacordado_delta2","mtarjeta_visa_descuentos_isNA_lag1","tpaquete4_delta2","cforex_delta2","cprestamos_prendarios_lag2","cpagodeservicios_lag1","mtarjeta_visa_consumo/mdescubierto_preacordado_delta2","minversion2_lag2","cplazo_fijo_lag1","Visa_madelantodolares_lag2","Master_mpagosdolares_delta2","cpagodeservicios","Visa_status_isNA_delta2","Visa_delinquency_isNA_delta1","minversion1_pesos_lag1","mv_status07_lag1","Master_mpagosdolares_delta1","tcuentas_lag2","cforex_sell_delta2","cforex_buy","tcuentas_delta2","Visa_status_delta1","cforex_sell_delta1","Visa_fultimo_cierre_isNA_delta2","Master_msaldodolares_delta2","Master_msaldodolares_delta1","mv_mpagosdolares_delta1","ccheques_depositados_rechazados_lag2","cprestamos_hipotecarios_lag2","Master_msaldodolares_isNA_delta2","ccajeros_propios_descuentos_delta2","active_quarter_lag2","Visa_Finiciomora_isNA_lag1","mtarjeta_master_descuentos_lag1","mpagodeservicios","minversion2_lag1","ctarjeta_master_descuentos_delta2","Master_msaldodolares_lag1","ctarjeta_master_descuentos_lag2","Master_Finiciomora_lag2","minversion1_pesos_lag2","mcaja_ahorro_adicional_lag1","ctarjeta_master_descuentos_delta1","Master_mpagado_lag1","cseguro_vivienda","Master_madelantopesos_isNA_delta2","mtarjeta_master_descuentos","Visa_Fvencimiento_isNA_delta2","ctarjeta_visa_descuentos_lag2","minversion2_delta1","tpaquete3_lag1","Master_mpagado_delta1","mprestamos_prendarios_delta1","mvr_madelantopesos","ctarjeta_master_descuentos","cplazo_fijo_delta1","ccajeros_propios_descuentos_lag1","mcaja_ahorro_adicional_delta2","ccheques_depositados_rechazados_lag1","Master_madelantopesos_lag2","cpagodeservicios_lag2","mcheques_depositados_rechazados","mv_status01/ctarjeta_visa_transacciones","tpaquete3_lag2","Master_fultimo_cierre_isNA_delta2","Visa_status_lag1","mpagodeservicios_delta1","Visa_status_isNA_delta1","mv_status05_lag2","mv_status03_lag2","mv_status05_lag1","mtarjeta_visa_descuentos_isNA_delta2","minversion1_pesos_delta2","ctarjeta_visa_transacciones/mv_status01","mcheques_emitidos_rechazados_delta1","Visa_mconsumospesos_isNA_lag1","Master_mconsumosdolares_isNA_delta2","mtarjeta_visa_descuentos_isNA_lag2","cinversion2_delta1","Visa_madelantopesos_isNA_delta2","ccheques_depositados_delta1","cpagodeservicios_delta2","Visa_cadelantosefectivo_delta1","ctarjeta_master_descuentos_lag1","cliente_edad_delta1","cprestamos_prendarios_delta1","mcheques_depositados_rechazados_delta1","Visa_Finiciomora_delta2","ccheques_emitidos_rechazados_delta1","mcheques_depositados_rechazados_lag2","cinversion2_lag2","mv_status06_lag2","cinversion1_lag1","Master_madelantodolares_isNA_lag2","cseguro_accidentes_personales_delta2","ccheques_depositados_delta2","Master_madelantodolares_isNA_delta2","Visa_Finiciomora_isNA_delta2","mv_status04_lag2","mtarjeta_master_descuentos_delta2","cseguro_auto","Master_madelantopesos_delta1","mv_status02_lag1","mtarjeta_master_descuentos_lag2","cseguro_vivienda_delta1","minversion1_pesos_delta1","Visa_mconsumosdolares_isNA_delta1","Master_mconsumospesos_isNA_delta1","Visa_madelantopesos_lag1","ccheques_emitidos_delta1","Master_cadelantosefectivo_lag2","Master_status_lag1","ccajeros_propios_descuentos_lag2","Master_madelantodolares_delta1","Visa_mconsumosdolares_isNA_delta2","cpagodeservicios_delta1","Visa_mfinanciacion_limite_isNA_delta2","Visa_madelantopesos_isNA_delta1","Visa_madelantodolares_isNA_delta2","mcheques_depositados_rechazados_delta2","Master_mpagospesos_isNA_delta2","Master_delinquency_lag2","Visa_mconsumospesos_isNA_delta2","cinversion2_lag1","cforex_sell_lag1","Master_madelantopesos_lag1","Master_madelantodolares_isNA_lag1","ccajeros_propios_descuentos","ctarjeta_master_debitos_automaticos_delta1","Master_mpagosdolares_isNA_lag2","Visa_madelantopesos","Visa_mconsumospesos_isNA_delta1","Visa_mpagosdolares_isNA_delta2","Master_delinquency_isNA_delta2","mv_cadelantosefectivo_delta1","mcaja_ahorro_adicional_delta1","mvr_madelantopesos_lag1","Visa_cadelantosefectivo_lag1","Visa_madelantodolares_delta1","mcajeros_propios_descuentos_lag2","Visa_status_lag2","active_quarter_lag1","cliente_vip","Visa_mfinanciacion_limite_isNA_lag2","Visa_fultimo_cierre_isNA_lag2","Master_mlimitecompra_isNA_delta2","Master_cadelantosefectivo_lag1","Master_madelantodolares_isNA_delta1","Visa_madelantopesos_delta2","cinversion1_lag2","mplazo_fijo_pesos_delta2","cpayroll2_trx_lag1","cseguro_auto_lag1","Master_Fvencimiento_isNA_delta1","active_quarter_delta2","Visa_cadelantosefectivo","Visa_cconsumos_isNA_delta1","cprestamos_prendarios_lag1","Visa_mpagosdolares_isNA_delta1","Master_cadelantosefectivo_delta1","active_quarter_delta1","mprestamos_hipotecarios_delta1","Master_Finiciomora_isNA_delta1","Master_mconsumototal_isNA_delta2","ctarjeta_visa_delta1","mv_madelantopesos_lag2","Visa_msaldopesos_isNA_delta2","tpaquete1","mv_status06_lag1","Visa_mconsumototal_isNA_delta2","Master_Fvencimiento_isNA_lag2","Master_madelantopesos_isNA_lag1","cseguro_vida_delta2","Master_cadelantosefectivo","mpayroll2_lag1","cforex_buy_lag2","ccheques_depositados_rechazados_delta1","cforex_buy_lag1","Visa_madelantopesos_delta1","Master_mpagosdolares_isNA_delta2","Visa_cadelantosefectivo_lag2","Visa_mconsumospesos_isNA_lag2","Visa_mlimitecompra_isNA_delta1","Master_madelantopesos_isNA_lag2","Master_mlimitecompra_isNA_delta1","Visa_mconsumototal_isNA_delta1","mtarjeta_visa_descuentos_isNA","cinversion1_delta1","Visa_madelantodolares_lag1","minversion1_dolares","mv_madelantopesos_delta1","mplazo_fijo_pesos_delta1","mprestamos_hipotecarios_delta2","mv_status02_lag2","mvr_madelantopesos_delta1","Master_delinquency_isNA_delta1","Visa_madelantopesos_lag2","cseguro_vida_delta1","Master_mconsumosdolares_isNA_delta1","Master_mpagosdolares_isNA_delta1","Visa_Finiciomora_isNA_lag2","Master_cconsumos_isNA_delta2","mtarjeta_visa_descuentos_isNA_delta1","mtarjeta_master_descuentos_isNA_delta1","Master_mconsumospesos_isNA_lag2","Master_delinquency_isNA_lag2","mdescubierto_preacordado_delta2","Master_madelantopesos_delta2")   #aqui van las columnas que se quieren eliminar

palancas$corregir <-  TRUE    # TRUE o FALSE

palancas$nuevasvars <-  FALSE  #si quiero hacer Feature Engineering manual

palancas$dummiesNA  <-  FALSE #La idea de Santiago Dellachiesa

palancas$lag1   <- FALSE    #lag de orden 1
palancas$delta1 <- FALSE    # campo -  lag de orden 1 
palancas$lag2   <- FALSE
palancas$delta2 <- FALSE
palancas$lag3   <- FALSE
palancas$delta3 <- FALSE
palancas$lag4   <- FALSE
palancas$delta4 <- FALSE
palancas$lag5   <- FALSE
palancas$delta5 <- FALSE
palancas$lag6   <- FALSE
palancas$delta6 <- FALSE

palancas$promedio3  <- FALSE  #promedio  de los ultimos 3 meses
palancas$promedio6  <- FALSE

palancas$minimo3  <- FALSE  #minimo de los ultimos 3 meses
palancas$minimo6  <- FALSE

palancas$maximo3  <- FALSE  #maximo de los ultimos 3 meses
palancas$maximo6  <- FALSE

palancas$ratiomax3   <- FALSE   #La idea de Daiana Sparta
palancas$ratiomean6  <- FALSE   #Un derivado de la idea de Daiana Sparta

palancas$tendencia6  <- FALSE    #Great power comes with great responsability


palancas$canaritosimportancia  <- TRUE  #si me quedo solo con lo mas importante de canaritosimportancia

palancas$Cocientes <- TRUE #Si uso cocientes entre variables seleccionadas. La eleccion de variables es manual. 

#escribo para saber cuales fueron los parametros
write_yaml(  palancas,  paste0( "./work/palanca_",  palancas$version  ,".yaml" ) )

#------------------------------------------------------------------------------

ReportarCampos  <- function( dataset )
{
  cat( "La cantidad de campos es ", ncol(dataset) , "\n" )
}
#------------------------------------------------------------------------------
#Agrega al dataset una variable que va de 1 a 12, el mes, para que el modelo aprenda estacionalidad

AgregarMes  <- function( dataset )
{
  dataset[  , mes := foto_mes %% 100 ]
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Elimina las variables que uno supone hace Data Drifting

DriftEliminar  <- function( dataset, variables )
{
  dataset[  , c(variables) := NULL ]
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#A las variables que tienen nulos, les agrega una nueva variable el dummy de is es nulo o no {0, 1}

DummiesNA  <- function( dataset )
{
  
  nulos  <- colSums( is.na(dataset[foto_mes==202101]) )  #cuento la cantidad de nulos por columna
  colsconNA  <- names( which(  nulos > 0 ) )
  
  dataset[ , paste0( colsconNA, "_isNA") :=  lapply( .SD,  is.na ),
           .SDcols= colsconNA]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Corrige poniendo a NA las variables que en ese mes estan da?adas

Corregir  <- function( dataset )
{
  #acomodo los errores del dataset
  
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
  
  dataset[ foto_mes==202006,  active_quarter   := NA ]
  dataset[ foto_mes==202006,  internet   := NA ]
  dataset[ foto_mes==202006,  mrentabilidad   := NA ]
  dataset[ foto_mes==202006,  mrentabilidad_annual   := NA ]
  dataset[ foto_mes==202006,  mcomisiones   := NA ]
  dataset[ foto_mes==202006,  mactivos_margen   := NA ]
  dataset[ foto_mes==202006,  mpasivos_margen   := NA ]
  dataset[ foto_mes==202006,  mcuentas_saldo   := NA ]
  dataset[ foto_mes==202006,  ctarjeta_debito_transacciones   := NA ]
  dataset[ foto_mes==202006,  mautoservicio   := NA ]
  dataset[ foto_mes==202006,  ctarjeta_visa_transacciones   := NA ]
  dataset[ foto_mes==202006,  mtarjeta_visa_consumo   := NA ]
  dataset[ foto_mes==202006,  ctarjeta_master_transacciones   := NA ]
  dataset[ foto_mes==202006,  mtarjeta_master_consumo   := NA ]
  dataset[ foto_mes==202006,  ccomisiones_otras   := NA ]
  dataset[ foto_mes==202006,  mcomisiones_otras   := NA ]
  dataset[ foto_mes==202006,  cextraccion_autoservicio   := NA ]
  dataset[ foto_mes==202006,  mextraccion_autoservicio   := NA ]
  dataset[ foto_mes==202006,  ccheques_depositados   := NA ]
  dataset[ foto_mes==202006,  mcheques_depositados   := NA ]
  dataset[ foto_mes==202006,  ccheques_emitidos   := NA ]
  dataset[ foto_mes==202006,  mcheques_emitidos   := NA ]
  dataset[ foto_mes==202006,  ccheques_depositados_rechazados   := NA ]
  dataset[ foto_mes==202006,  mcheques_depositados_rechazados   := NA ]
  dataset[ foto_mes==202006,  ccheques_emitidos_rechazados   := NA ]
  dataset[ foto_mes==202006,  mcheques_emitidos_rechazados   := NA ]
  dataset[ foto_mes==202006,  tcallcenter   := NA ]
  dataset[ foto_mes==202006,  ccallcenter_transacciones   := NA ]
  dataset[ foto_mes==202006,  thomebanking   := NA ]
  dataset[ foto_mes==202006,  chomebanking_transacciones   := NA ]
  dataset[ foto_mes==202006,  ccajas_transacciones   := NA ]
  dataset[ foto_mes==202006,  ccajas_consultas   := NA ]
  dataset[ foto_mes==202006,  ccajas_depositos   := NA ]
  dataset[ foto_mes==202006,  ccajas_extracciones   := NA ]
  dataset[ foto_mes==202006,  ccajas_otras   := NA ]
  dataset[ foto_mes==202006,  catm_trx   := NA ]
  dataset[ foto_mes==202006,  matm   := NA ]
  dataset[ foto_mes==202006,  catm_trx_other   := NA ]
  dataset[ foto_mes==202006,  matm_other   := NA ]
  dataset[ foto_mes==202006,  ctrx_quarter   := NA ]
  dataset[ foto_mes==202006,  tmobile_app   := NA ]
  dataset[ foto_mes==202006,  cmobile_app_trx   := NA ]
  
  
  dataset[ foto_mes==202010,  internet  := NA ]
  dataset[ foto_mes==202011,  internet  := NA ]
  dataset[ foto_mes==202012,  internet  := NA ]
  dataset[ foto_mes==202101,  internet  := NA ]
  
  dataset[ foto_mes==202009,  tmobile_app  := NA ]
  dataset[ foto_mes==202010,  tmobile_app  := NA ]
  dataset[ foto_mes==202011,  tmobile_app  := NA ]
  dataset[ foto_mes==202012,  tmobile_app  := NA ]
  dataset[ foto_mes==202101,  tmobile_app  := NA ]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Esta es la parte que los alumnos deben desplegar todo su ingenio

AgregarVariables  <- function( dataset )
{
  #INICIO de la seccion donde se deben hacer cambios con variables nuevas
  #se crean los nuevos campos para MasterCard  y Visa, teniendo en cuenta los NA's
  #varias formas de combinar Visa_status y Master_status
  dataset[ , mv_status01       := pmax( Master_status,  Visa_status, na.rm = TRUE) ]
  
  
  
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
  
  ReportarCampos( dataset )
}


Cocientes  <- function( dataset )
{ 
  #creo vector con variables flash 
  
  tipom <- c("mcaja_ahorro","mdescubierto_preacordado_delta2","mtarjeta_visa_consumo","cpayroll_trx","mv_mpagospesos","mv_status01","ctrx_quarter_lag1","ctarjeta_visa_transacciones","mcaja_ahorro_lag1","ctarjeta_debito_transacciones","mpayroll","mvr_mpagospesos","mprestamos_personales_delta2","mcuenta_corriente","mprestamos_personales","cpayroll_trx_lag1")
  
  #creo cocientes entre tipom y tipom
  
  for (vcol in tipom){
    for (vcol2 in tipom) {
      if(vcol != vcol2){
        dataset[, paste0(vcol, "/",vcol2) := get(vcol)/get(vcol2)]
      }
    }
  }
}


#------------------------------------------------------------------------------
#esta funcion supone que dataset esta ordenado por   <numero_de_cliente, foto_mes>
#calcula el lag y el delta lag

Lags  <- function( dataset, cols, nlag, deltas )
{
  
  sufijo  <- paste0( "_lag", nlag )
  
  dataset[ , paste0( cols, sufijo) := shift(.SD, nlag, NA, "lag"), 
           by= numero_de_cliente, 
           .SDcols= cols]
  
  #agrego los deltas de los lags, con un "for" nada elegante
  if( deltas )
  {
    sufijodelta  <- paste0( "_delta", nlag )
    
    for( vcol in cols )
    {
      dataset[,  paste0(vcol, sufijodelta) := get( vcol)  - get(paste0( vcol, sufijo))]
    }
  }
  
  ReportarCampos( dataset )
}


#------------------------------------------------------------------------------
#calcula el promedio de los ultimos  nhistoria meses

Promedios  <- function( dataset, cols, nhistoria )
{
  
  sufijo  <- paste0( "_avg", nhistoria )
  
  dataset[ , paste0( cols, sufijo) := frollmean(x=.SD, n=nhistoria, na.rm=TRUE, algo="fast", align="right"), 
           by= numero_de_cliente, 
           .SDcols= cols]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#calcula el minimo de los ultimos  nhistoria meses

Minimos  <- function( dataset, cols, nhistoria )
{
  
  sufijo  <- paste0( "_min", nhistoria )
  
  dataset[ , paste0( cols, sufijo) := frollapply(x=.SD, FUN="min", n=nhistoria, align="right"), 
           by= numero_de_cliente, 
           .SDcols= cols]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#calcula el maximo de los ultimos  nhistoria meses

Maximos  <- function( dataset, cols, nhistoria )
{
  
  sufijo  <- paste0( "_max", nhistoria )
  
  dataset[ , paste0( cols, sufijo) := frollapply(x=.SD, FUN="max", n=nhistoria, na.rm=TRUE, align="right"), 
           by= numero_de_cliente, 
           .SDcols= cols]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#calcula  el ratio entre el valor actual y el maximo de los ultimos nhistoria meses

RatioMax  <- function( dataset, cols, nhistoria )
{
  sufijo  <- paste0( "_rmax", nhistoria )
  
  dataset[ , paste0( cols, sufijo) := .SD/ frollapply(x=.SD, FUN="max", n=nhistoria, na.rm=TRUE, align="right"), 
           by= numero_de_cliente, 
           .SDcols= cols]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#calcula  el ratio entre el valor actual y el promedio de los ultimos nhistoria meses

RatioMean  <- function( dataset, cols, nhistoria )
{
  sufijo  <- paste0( "_rmean", nhistoria )
  
  dataset[ , paste0( cols, sufijo) := .SD/frollapply(x=.SD, FUN="mean", n=nhistoria, na.rm=TRUE, align="right"), 
           by= numero_de_cliente, 
           .SDcols= cols]
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------

#se calculan para los 6 meses previos el minimo, maximo y tendencia calculada con cuadrados minimos
#la formual de calculo de la tendencia puede verse en https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/10%3A_Correlation_and_Regression/10.04%3A_The_Least_Squares_Regression_Line
#para la max?ma velocidad esta funcion esta escrita en lenguaje C, y no en la porqueria de R o Python

Rcpp::cppFunction('NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde ) 
{
  // [[Rcpp::plugins(openmp)]]
  /* Aqui se cargan los valores para la regresion */
  double  x[100] ;
  double  y[100] ;

  int n = pcolumna.size();
  NumericVector out( n );


  //#if defined(_OPENMP)
  //#pragma omp parallel for
  //#endif
  for(int i = 0; i < n; i++)
  {
    int  libre    = 0 ;
    int  xvalor   = 1 ;

    for( int j= pdesde[i]-1;  j<=i; j++ )
    {
       double a = pcolumna[j] ;

       if( !R_IsNA( a ) ) 
       {
          y[ libre ]= a ;
          x[ libre ]= xvalor ;
          libre++ ;
       }

       xvalor++ ;
    }

    /* Si hay al menos dos valores */
    if( libre > 1 )
    {
      double  xsum  = x[0] ;
      double  ysum  = y[0] ;
      double  xysum = xsum * ysum ;
      double  xxsum = xsum * xsum ;
      double  vmin  = y[0] ;
      double  vmax  = y[0] ;

      for( int h=1; h<libre; h++)
      { 
        xsum  += x[h] ;
        ysum  += y[h] ; 
        xysum += x[h]*y[h] ;
        xxsum += x[h]*x[h] ;

        if( y[h] < vmin )  vmin = y[h] ;
        if( y[h] > vmax )  vmax = y[h] ;
      }

      out[ i ]  =  (libre*xysum - xsum*ysum)/(libre*xxsum -xsum*xsum) ;
    }
    else
    {
      out[ i ]  =  NA_REAL ; 
    }
  }

  return  out;
}')

#------------------------------------------------------------------------------
#calcula la tendencia de las variables cols de los ultimos 6 meses
#la tendencia es la pendiente de la recta que ajusta por cuadrados minimos

Tendencia  <- function( dataset, cols )
{
  #Esta es la cantidad de meses que utilizo para la historia
  ventana_regresion  <- 6
  
  last  <- nrow( dataset )
  
  #creo el vector_desde que indica cada ventana
  #de esta forma se acelera el procesamiento ya que lo hago una sola vez
  vector_ids   <- dataset$numero_de_cliente
  
  vector_desde  <- seq( -ventana_regresion+2,  nrow(dataset)-ventana_regresion+1 )
  vector_desde[ 1:ventana_regresion ]  <-  1
  
  for( i in 2:last )  if( vector_ids[ i-1 ] !=  vector_ids[ i ] ) {  vector_desde[i] <-  i }
  for( i in 2:last )  if( vector_desde[i] < vector_desde[i-1] )  {  vector_desde[i] <-  vector_desde[i-1] }
  
  for(  campo  in   cols )
  {
    nueva_col     <- fhistC( dataset[ , get(campo) ], vector_desde ) 
    
    dataset[ , paste0( campo, "_tend") := nueva_col[ (0*last +1):(1*last) ]  ]
  }
  
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
  
  gan  <- mean( tbl[ 1:500,  gan_acum] )  #meseta de tama?o 500
  
  pos_meseta  <- tbl[ 1:500,  median(posicion)]
  VPOS_CORTE  <<- c( VPOS_CORTE, pos_meseta )
  
  return( list( "name"= "ganancia", 
                "value"=  gan,
                "higher_better"= TRUE ) )
}
#------------------------------------------------------------------------------
#Elimina del dataset las variables que estan por debajo de la capa geologica de canaritos

CanaritosImportancia  <- function( dataset )
{
  
  gc()
  dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]
  
  for( i  in 1:(ncol(dataset)/5))  dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset))]
  
  campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01" ) )
  
  azar  <- runif( nrow(dataset) )
  entrenamiento  <-  dataset[ , foto_mes>= 202001 &  foto_mes<= 202010 &  foto_mes!=202006 & ( clase01==1 | azar < 0.10 ) ]
  
  dtrain  <- lgb.Dataset( data=    data.matrix(  dataset[ entrenamiento==TRUE, campos_buenos, with=FALSE]),
                          label=   dataset[ entrenamiento==TRUE, clase01],
                          weight=  dataset[ entrenamiento==TRUE, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)] )
  
  dvalid  <- lgb.Dataset( data=    data.matrix(  dataset[ foto_mes==202011, campos_buenos, with=FALSE]),
                          label=   dataset[ foto_mes==202011, clase01],
                          weight=  dataset[ foto_mes==202011, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)] )
  
  
  param <- list( objective= "binary",
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
                 force_row_wise= TRUE,    #para que los alumnos no se atemoricen con tantos warning
                 learning_rate= 0.02, 
                 feature_fraction= 0.50,
                 min_data_in_leaf= 4000,
                 num_leaves= 600,
                 early_stopping_rounds= 200 )
  
  modelo  <- lgb.train( data= dtrain,
                        valids= list( valid= dvalid ),
                        eval= fganancia_lgbm_meseta,
                        param= param,
                        verbose= -100 )
  
  tb_importancia  <- lgb.importance( model= modelo )
  tb_importancia[  , pos := .I ]
  
  fwrite( tb_importancia, file="./work/impo.txt", sep="\t" )
  
  umbral  <- tb_importancia[ Feature %like% "canarito", median(pos) - sd(pos) ]
  col_inutiles  <- tb_importancia[ pos >= umbral | Feature %like% "canarito",  Feature ]
  
  for( col in col_inutiles )
  {
    dataset[  ,  paste0(col) := NULL ]
  }
  
  rm( dtrain, dvalid )
  gc()
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------

correr_todo  <- function( palancas )
{
  #cargo el dataset ORIGINAL
  dataset  <- fread( "./datasets/dataset_epic_v006.csv.gz")
  
  setorder(  dataset, numero_de_cliente, foto_mes )  #ordeno el dataset
  
  AgregarMes( dataset )  #agrego el mes del a?o
  
  if( palancas$dummiesNA )  DummiesNA( dataset )  #esta linea debe ir ANTES de Corregir  !!
  
  if( palancas$corregir )  Corregir( dataset )  #esta linea debe ir DESPUES de  DummiesNA
  
  if( palancas$nuevasvars )  AgregarVariables( dataset )
  
  cols_analiticas  <- setdiff( colnames(dataset),  c("numero_de_cliente","foto_mes","mes","clase_ternaria") )
  
  if( palancas$lag1 )   Lags( dataset, cols_analiticas, 1, palancas$delta1 )
  if( palancas$lag2 )   Lags( dataset, cols_analiticas, 2, palancas$delta2 )
  if( palancas$lag3 )   Lags( dataset, cols_analiticas, 3, palancas$delta3 )
  if( palancas$lag4 )   Lags( dataset, cols_analiticas, 4, palancas$delta4 )
  if( palancas$lag5 )   Lags( dataset, cols_analiticas, 5, palancas$delta5 )
  if( palancas$lag6 )   Lags( dataset, cols_analiticas, 6, palancas$delta6 )
  
  
  if( palancas$promedio3 )  Promedios( dataset, cols_analiticas, 3 )
  if( palancas$promedio6 )  Promedios( dataset, cols_analiticas, 6 )
  
  if( palancas$minimo3 )  Minimos( dataset, cols_analiticas, 3 )
  if( palancas$minimo6 )  Minimos( dataset, cols_analiticas, 6 )
  
  if( palancas$maximo3 )  Maximos( dataset, cols_analiticas, 3 )
  if( palancas$maximo6 )  Maximos( dataset, cols_analiticas, 6 )
  
  if(palancas$ratiomax3)  RatioMax(  dataset, cols_analiticas, 3) #La idea de Daiana Sparta
  if(palancas$ratiomean6) RatioMean( dataset, cols_analiticas, 6) #Derivado de la idea de Daiana Sparta
  
  
  if( palancas$tendencia6 )  Tendencia( dataset, cols_analiticas)
  
  if( palancas$Cocientes ) Cocientes( dataset )
  
  if( length(palancas$variablesdrift) > 0 )   DriftEliminar( dataset, palancas$variablesdrift )
  
  if( palancas$canaritosimportancia )  CanaritosImportancia( dataset )
  
  
  
  #dejo la clase como ultimo campo
  nuevo_orden  <- c( setdiff( colnames( dataset ) , "clase_ternaria" ) , "clase_ternaria" )
  setcolorder( dataset, nuevo_orden )
  
  #Grabo el dataset
  fwrite( dataset,
          paste0( "./datasets/dataset_epic_", palancas$version, ".csv.gz" ),
          logical01 = TRUE,
          sep= "," )
  
}
#------------------------------------------------------------------------------

#Aqui empieza el programa


correr_todo( palancas )


quit( save="no" )


