import * as tf from '@tensorflow/tfjs-node-gpu';
import { saveAs } from 'file-saver';

const seed = 64;
const init = tf.initializers.heUniform(seed);

export class MaskLayer extends tf.layers.Layer {
  constructor(config) {
  super(config);
    this.seed = config.seed;
    this.temp = config.temp;
    this.dropout = config.dropout;
  }
  build(inputShape){
    const feat = inputShape[1];
    this.logits = this.addWeight('logits', [1, feat], 'float32', tf.initializers.constant({value:0.0}));
  }
  computeOutputShape(inputShape) { 
    return [3, null, 1, 1, this.ndim]
  }
  call(inputs, kwargs) {
    let x = inputs[0]
    return tf.tidy(() => {
      let nanmask = tf.zerosLike(x)
          .where(tf.isNaN(x), tf.onesLike(x))
          .expandDims(1).expandDims(2); // 0 if missing, 1 if present (b, 1, 1, f)
      let eps = 1e-8;
      let noise = tf.randomUniform(x.shape, 0, 1, 'float32', this.seed);
      let probs = this.logits.read().sigmoid();
      let drop = tf.sigmoid(tf.div(tf.log(probs.add(eps))
                .sub(tf.log(probs.mul(-1.0).add(1.0).add(eps)))
                .add(tf.log(noise.add(eps)))
                .sub(tf.log(noise.mul(-1.0).add(1.0).add(eps))),
                this.temp)).mul(-1.0).add(1.0); // 1 if keeping, 0 if dropping
      let randmask = tf.where(x.mul(tf.scalar(this.dropout)).equal(x), drop, 1.0)
          .expandDims(1).expandDims(2); // if dropout is true then apply drop, else just keep all
      let mask = nanmask.mul(randmask)
      return tf.stack([mask, nanmask, randmask])
    });
  }
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      seed: this.seed,
      temp: this.temp,
      dropout: this.dropout
    });
    return config;
  }
  static get className() {
    return 'MaskLayer';
  }
}

export class QueryLayer extends tf.layers.Layer {
  constructor(config) {
  super(config);
    this.ndim=config.ndim;
  }
  build(inputShape){
    const feat = inputShape[1];
    this.query = this.addWeight('query', [1, feat, this.ndim], 'float32', init);
  }
  computeOutputShape(inputShape) { 
    return inputShape; 
  }
  call(input) {
    return tf.tidy(() => tf.ones([input[0].shape[0], 1, 1])
          .mul(this.query.read()));
  }
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      ndim: this.ndim
    });
    return config;
  }
  static get className() {
    return 'QueryLayer';
  }
}

export class EmbedLayer extends tf.layers.Layer {
  constructor(config) {
  super(config);
    this.ndim=config.ndim;
  }
  build(inputShape){
    const feat = inputShape[1];
    this.shift = this.addWeight('sw', [1, feat, this.ndim], 'float32', init);
    this.scale = this.addWeight('bw', [1, feat, 1], 'float32', init);
  }
  computeOutputShape(inputShape) { 
    return [... inputShape, ... [this.ndim]]; 
  }
  call(input) {
    return tf.tidy(() => input[0].expandDims(2)
          .add(this.shift.read()).mul(this.scale.read()));
  }
  reverse(input){
    return tf.tidy(() => input.mul(this.scale.read().reciprocal()).sub(this.shift.read())
                          .mean(-1));
  }
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      ndim: this.ndim
    });
    return config;
  }
  static get className() {
    return 'EmbedLayer';
  }
}

export class BijectorLayer extends tf.layers.Layer {
  constructor(config) {
  super(config);
  this.layer = config.layer;
  }
  build(inputShape){
    const ndim = inputShape[2];
    const feat = inputShape[1];
    const idx = ndim / 2 | 0;
    this.sw = this.addWeight('sw', [feat, idx, ndim-idx], 'float32', init);
    this.sb = this.addWeight('bw', [feat, 1, ndim-idx], 'float32', init);
    this.tw = this.addWeight('tw', [feat, idx, ndim-idx], 'float32', init);
    this.tb = this.addWeight('tb', [feat, 1, ndim-idx], 'float32', init);
    this.ndim=ndim;
    this.feat=feat;
    this.idx=idx;
  }
  computeOutputShape(inputShape) { 
    return inputShape; 
  }
  call(input, kwargs) {
    const x = input[0].expandDims(2);
    this.invokeCallHook(input, kwargs);
    let forward = kwargs['forward'];
    if (typeof forward == 'undefined'){forward = true};
    if (forward){
      if (this.layer % 2 == 0) {
        let out = tf.stack(x.unstack().map(tensor => { 
        const [x1, x2] = tf.split(tensor, [this.idx, this.ndim-this.idx], 2);
        const y2 = x2.add(x1.matMul(this.tw.read()).add(this.tb.read()).softplus())
          .mul(x1.matMul(this.sw.read()).add(this.sb.read()).tanh().exp());
        return x1.concat(y2, 2)
        }));
        return out.squeeze(2)
      } else {
          // const x1, x2;
          let out = tf.stack(x.unstack().map(tensor => { 
          const [x1, x2] = tf.split(tensor, [this.ndim-this.idx, this.idx], 2);
          const y1 = x1.add(x2.matMul(this.tw.read()).add(this.tb.read()).softplus())
            .mul(x2.matMul(this.sw.read()).add(this.sb.read()).tanh().exp());
          return y1.concat(x2, 2)
          }));
          return out.squeeze(2)
      }
    }
    else {
        if (this.layer % 2 == 0) {
          let out = tf.stack(x.unstack().map(tensor => {
          const [x1, x2] = tf.split(tensor, [this.idx, this.ndim-this.idx], 2);
          let y2 = x2.mul(x1.matMul(this.sw.read()).add(this.sb.read()).tanh().mul(-1).exp())
            .sub(x1.matMul(this.tw.read()).add(this.tb.read()).softplus());
          return x1.concat(y2, 2)
          }));
          return out.squeeze(2)
        } else {
          let out = tf.stack(x.unstack().map(tensor => {
          const [x1, x2] = tf.split(tensor, [this.ndim-this.idx, this.idx], 2);
          let y1 = x1.mul(x2.matMul(this.sw.read()).add(this.sb.read()).tanh().mul(-1).exp())
            .sub(x2.matMul(this.tw.read()).add(this.tb.read()).softplus());
          return y1.concat(x2, 2)
          }));
          return out.squeeze(2)
        }
      }
    }
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      layer: this.layer
    });
    return config;
  }
  static get className() {
    return 'BijectorLayer';
  }
}

export class AttentionLayer extends tf.layers.Layer {
  constructor(config) {
  super(config);
  this.heads=config.heads;
  }
  build(inputShape){
    const ndim = inputShape[0][2];
    const feat = inputShape[0][1];
    this.qw = this.addWeight('qw', [ndim, ndim*this.heads], 'float32', init);
    this.qb = this.addWeight('qb', [ndim*this.heads], 'float32', init);
    this.kw = this.addWeight('kw', [ndim, ndim*this.heads], 'float32', init);
    this.kb = this.addWeight('kb', [ndim*this.heads], 'float32', init);
    this.vw = this.addWeight('vw', [ndim, ndim*this.heads], 'float32', init);
    this.vb = this.addWeight('vb', [ndim*this.heads], 'float32', init);
    this.ow = this.addWeight('ow', [ndim*this.heads, ndim], 'float32', init);
    this.ob = this.addWeight('ob', [ndim], 'float32', init);

    this.l1w = this.addWeight('l1w', [ndim, ndim], 'float32', init);
    this.l1b = this.addWeight('l1b', [ndim], 'float32', init);
    this.l2w = this.addWeight('l2w', [ndim, ndim], 'float32', init);
    this.l2b = this.addWeight('l2b', [ndim], 'float32', init);

    this.ndim=ndim;
    this.feat=feat;
  }
  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1]]; 
  }
  call(inputs) {
    return tf.tidy(() => {

    const query = inputs[0];
    let qs = Array.from(query.shape);
    qs.push(this.heads);
    let q = tf.stack(query.unstack(1).map(
      tensor => tensor.matMul(this.qw.read()).add(this.qb.read())
    ), 1).reshape(qs).transpose([0,3,1,2]); // (b, h, f1, d)

    const key = inputs[1];
    const mask = inputs[2].slice([0], [1]).squeeze(0);
    let ks = Array.from(key.shape);
    ks.push(this.heads);
        
    let k = tf.stack(key.unstack(1).map(
      tensor => tensor.matMul(this.kw.read()).add(this.kb.read())
    ), 1).reshape(ks).transpose([0,3,1,2]); // (b, h, f1, d)

    let v = tf.stack(key.unstack(1).map(
      tensor => tensor.matMul(this.vw.read()).add(this.vb.read())
    ), 1).reshape(ks).transpose([0,3,1,2]); // (b, h, f1, d)

    let qk = q.matMul(k, false, true).softplus(); // (b, h, f1, f2)
    let attn = qk.mul(mask).div(qk.sum(-1, true).add(1e-8)); // (b, h, f1, f2)

    let out = attn.matMul(v).transpose([0,2,3,1])
      .reshape([-1, qs[1], this.heads*this.ndim]); // (b, f1, d*h)

    out = tf.stack(out.unstack(1).map(
      tensor => tensor.matMul(this.ow.read()).add(this.ob.read())
          .matMul(this.l1w.read()).add(this.l1b.read()).softplus()
          .matMul(this.l2w.read()).add(this.l2b.read())
    ), 1); // (b, h, f1, d)
    return [out, key]
    });
  }
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, {
      heads: this.heads
    });
    return config;
  }
  static get className() {
    return 'AttentionLayer';
  }
}

tf.serialization.registerClass(MaskLayer);
tf.serialization.registerClass(QueryLayer);
tf.serialization.registerClass(EmbedLayer);
tf.serialization.registerClass(BijectorLayer);
tf.serialization.registerClass(AttentionLayer);

export class AttentionModel{
  constructor(config){
    this.init(config);
  }
  forward(x){
    const [bij, out, enckey, masks, embed] = this.model.predict(x);
    const bij2 = this.bijector.reduce((pv, cv, i) => cv.apply([pv], {forward:true}), embed);
    const invert = this.bijector.reverse().reduce((pv, cv, i) => cv.apply([pv], {forward:false}), out);
    const recon = this.embed.reverse(invert);
    console.log('d', x.shape, x.toString());
    //console.log('e', embed.shape, embed.toString());
    //console.log('bij', bij.shape, bij.toString());
    //console.log('bij2', bij2.shape, bij2.toString());
    //console.log('ehat', invert.shape, invert.toString());
    console.log('recon', recon.shape, recon.toString());
    //console.log('mask', masks.slice(2,1).shape, masks.slice(2,1).toString());
    return out, invert
  }

  async train(epochs, ds, vis){
    let lossSave = [];
    let history = {
      values:[lossSave],
      series:['MSE']
    };
    const options = {
        xLabel: 'Epoch',
        yLabel: 'MSE',
        seriesColors: ['tomato']
      };
    const surface = {
        name: 'Training Charts',
        tab: 'Training'
      };
    const optimizer = tf.train.adam(0.001);
    for (let epoch = 0; epoch < epochs; epoch++) {
      let lossAvg = [];
      await ds.forEachAsync((xs) => {
        optimizer.minimize(() => {
          const results = this.model.predict(xs);
          const out = results[1];
          const bij = results[0];
          const invmask = results[3].slice(2,1).add(-1).mul(-1);
          const nanmask = results[3].slice(1,1);
          const mask = invmask.mul(nanmask).squeeze([0, 2, 3]).expandDims(2);
          const loss = tf.square(bij.sub(out)).mul(mask).mean();
          // loss.data().then(l => console.log('Loss', l[0]));
          loss.data().then(l =>  lossAvg.push(l[0]));
          return loss;
        });
      });
      let avg = lossAvg.reduce((a, b) => a + b, 0) / lossAvg.length;
      lossSave.push({x:epoch, y:avg});
      if (vis){tfvis.render.linechart(surface, history, options)};
      console.log('Epoch', epoch, ':', avg);
    }
  }

  async distance(){

  }

  async save(db){
    let savejson = {
      config: this.config,
      weights: []
    };
    this.model.layers.forEach(layer => {
      let w = layer.getWeights();
      w = w.map(v => [v.dataSync(), v.shape]);
      savejson.weights.push(w);
    });
    // saveAs(JSON.stringify(savejson), dest);
    const id = await db.models.add({
      datetime: Date.now(),
      name: this.name,
      config: savejson.config,
      weights: savejson.weights 
    });
  }
  async save_alt(){
    await this.model.save('indexeddb://full')
  }
  async load_alt(){
    this.model = await tf.loadLayersModel('indexeddb://full');
    this.bijector = await this.model.layers.filter(l => l.name.includes('bijector'));
    this.embed = await this.model.layers.filter(l => l.name.includes('embed'))[0];
    console.log("model loaded");
  }

  async load(db, name){
      const query = await db.models.where("name").equals(name).first();
      if (query){
        console.log(query);
        // await this.init(query.config);
        console.log(this.model.layers);
        this.model.layers.forEach((layer, idx) => {
          let w = query.weights[idx];
          w = w.map(v => tf.tensor(v[0], v[1], 'float32'));
          layer.setWeights(w);
        });
        console.log("created model and loaded weights");
      } else {
        console.log("no entry in DB")
      }
      // w = w.map(v => tf.tensor(v));
      // console.log(w);
  }

  init(config){
    this.config = config;
    this.feat = config.feat;
    this.ndim = config.ndim;
    this.heads = config.heads;
    this.encLayers = config.encLayers;
    this.decLayers = config.decLayers;
    this.bijectLayers = config.bijectLayers;
    this.seed = config.seed;
    this.temp = config.temp;
    this.name = config.name;
    this.datadict = config.datadict;

    // construct base model
    const input = tf.input({shape: [this.feat]});
    const masks = new MaskLayer({seed:this.seed, temp:this.temp, dropout:true, name:'MaskLayer'}).apply(input);
    const embed = new EmbedLayer({ndim:this.ndim}).apply(input);
    const bijection = Array(this.bijectLayers).fill().map(
      (x, i) => new BijectorLayer({layer:i}));
    const encoder = Array(this.encLayers).fill().map(
      (x, i) => new AttentionLayer({heads:this.heads}));
    const decoder = Array(this.decLayers).fill().map(
      (x, i) => new AttentionLayer({heads:this.heads}));
    const bij = bijection.reduce((pv, cv, i) => cv.apply(pv, {forward:true}), embed);
    const [enckey, del1] = encoder.reduce((pv, cv, i) => cv.apply([pv[0], pv[0], masks]), [bij, bij]);
    const query = new QueryLayer({ndim:this.ndim}).apply(embed);
    const [out, del2] = decoder.reduce((pv, cv, i) => cv.apply([pv[0], pv[1], masks]), [query, enckey]);
    this.model = tf.model({inputs:input, outputs:[bij, out, enckey, masks, embed]});
    this.bijector = bijection;
    this.embed = this.model.layers.filter(l => l.name.includes('embed'))[0];
    this.maskidx = this.model.layers.findIndex((el) => el.name == 'MaskLayer')
  }
}

