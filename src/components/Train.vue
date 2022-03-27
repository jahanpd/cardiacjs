
<template>
  <div class="flex flex-row justify-center p-5 flex-wrap md:flex-nowrap">
      <h1>Training</h1>
  </div>
  <div class="flex flex-row justify-center p-5 flex-wrap md:flex-nowrap">
     <button 
      class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      v-on:click="toggle()"
      >
      Toggle TF Vis JS
    </button>
  </div>
 
    </template>
<script>
// import functions
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { AttentionModel, BijectorLayer } from '../assets/model/model';
import { db } from '../assets/db';

await tf.setBackend('cpu');

let vis = tfvis.visor();
const feat = 4;
const data = tf.randomNormal([20, feat], 0, 1, 'float32', 64);
console.log(data.shape);
const ds = tf.data.array(data.clone().arraySync()).repeat(10).batch(10);

// const model = createModel(17, feat, 4, 5, 3, 3);
let model = await new AttentionModel({
    name: 'testing',
    feat:feat,
    ndim:10,
    heads:2,
    bijectLayers:5,
    encLayers:2,
    decLayers:2,
    seed:64,
    temp:0.05,
})

// model.model.summary();
// await model.load(db,"testing");
model.save_alt();
model.load_alt();
// let out = await model.forward(data);
model.train(10, ds)


async function gradienttest(){
  const {value, grads} = tf.variableGrads(
    () => {
      let out = model.model.predict(data);
      return out[1].clone().sum()
    }
  );
  console.log("grads", grads);
}
// gradienttest()

export default {
  name: 'Landing',
  data: () => ({
    vis:vis,
    config:{
      name: 'testing',
      feat:feat,
      ndim:17,
      heads:4,
      bijectLayers:5,
      encLayers:3,
      decLayers:3,
      seed:64,
      temp:0.05,
    },
    model: model,
    db: db
  }),
  methods: {
    toggle(){
      this.vis.toggle();
    },
    async createModel(){
      this.model = await new AttentionModel(this.config);
      console.log('model created');
    },
    clearAll(){
      db.delete()
    }

  }
}
</script>
