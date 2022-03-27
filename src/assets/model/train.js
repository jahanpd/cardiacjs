import * as tf from '@tensorflow/tfjs'

export async function unsupervised(data, model){
  const optimizer = tf.train.sgd(0.1 /* learningRate */);
  // Train for 5 epochs.
  let seed = 1;
  for (let epoch = 0; epoch < 5; epoch++) {
    await data.forEachAsync((xs) => {
      optimizer.minimize(() => {
        const res = model.predict(xs);
        seed++; 
        const invmask = res[2].add(-1).mul(-1);
        const loss = res[0];
        // const loss = res.out.sub(res.bijection).pow(2).sum(-1)
        //   .mul(invmask).mul(res.nanmask).mean(-1);
        console.log('Loss', loss.arraySync());
        return loss;
      });
      console.log(grads);
    });
    console.log('Epoch', epoch);
  }
}
