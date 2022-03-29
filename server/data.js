import * as tf from '@tensorflow/tfjs';
import * as dfd from "danfojs-node";
import fetch from 'node-fetch';

let path = 'http://localhost:1234/full_data.csv';
let path2 = 'http://localhost:1234/app/static/input_data.json'

import * as  fs from 'fs'

let predictors = ['DOA', 'OpID', 'DOP', 'opno', 'AdmitID', 'PatID', 'ICU', 'VENT', 'TP', 'PROCNO', 'AGE', 'Sex', 'Race1', 'Insur', 'SMO_H', 'SMO_C', 'DB', 'DB_CON', 'HCHOL', 'PRECR', 'DIAL', 'TRANS', 'HG', 'HYT', 'CBVD', 'CBVD_T', 'CVA_W', 'CART', 'PVD', 'LD', 'LD_T', 'IE', 'IE_T', 'IMSRX', 'MI', 'MI_T', 'MI_W', 'CCS', 'ANGRXG', 'ANGRXH', 'ANGRXC', 'ANG_T', 'CHF', 'CHF_C', 'NYHA', 'SHOCK', 'RESUS', 'ARRT', 'ARRT_A', 'ARRT_AT', 'ARRT_H', 'ARRT_V', 'PACE', 'MEDIN', 'MEDAC', 'MEDST', 'MED_ASP', 'MED_CLOP', 'MED_TICA', 'POP', 'PCS', 'PTAVR', 'PTCA', 'PTCA_ADM', 'PTCA_H', 'CATH', 'EF', 'EF_EST', 'LMD', 'DISVES', 'BMI', 'eGFR', 'PROC', 'STAT', 'DTCATH', 'TRAUMA', 'TUMOUR', 'CT', 'LAA', 'AO', 'AOP', 'MIN', 'CPB', 'CCT', 'PERF', 'MINHT', 'IABP', 'IABP_W', 'IABP_I', 'ECMO', 'ECMO_W', 'ECMO_I', 'VAD', 'VAD_W', 'VAD_I', 'IOTOE', 'ANTIFIB', 'ANTIFIB_T', 'IDGCA', 'ITA', 'DAN_AC', 'DANV', 'DAN', 'AOPROC', 'AOPATH', 'MIPROC', 'MIPATH', 'TRPROC', 'TRPATH', 'PUPROC', 'PUPATH', 'AOPLN', 'RBC', 'RBCUnit', 'NRBC', 'PlateUnit', 'NovoUnit', 'CryoUnit', 'FFPUnit', 'DRAIN_4', 'PINT'];

let outcomes = ['REICU', 'REINT', 'RTT', 'NRF', 'HAEMOFIL', 'NARRT', 'VENT_P', 'PUPNU', 'INFDS', 'MORT30', 'READ'];

let response = await fetch(path2);
let datadictOG = await response.json();

let datadict = predictors.concat(outcomes).map(item => {
  let filt = datadictOG.filter(x => x.colname == item);
  if (filt.length > 0){
    let out = filt[0];
    out['required'] = true;
    return out
  } else {
    return Object({
        "name": "full",
        "colname": item,
        "type": "",
        "class":"",
        "colno": 0,
        "values": [],
        "val_label": "continuous categorical",
        "units": "",
        "timing": 0,
        "level": 1,
        "parent": "NaN",
        "parent_val": "NaN",
        "nan_count": 0,
        "child": [],
        "required":false
      })
  }
}
);
// fs.writeFileSync('./src/assets/datadict.json', JSON.stringify(datadict));

let df = await dfd.readCSV(path);
console.log(df.shape);
console.log(df['DB_CON'].isNa().sum());

export function createData(datapath, datadictpath){
  const csvDataset = tf.data.csv(datapath);
  // prep datadict
  

}
