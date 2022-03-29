import * as tf from '@tensorflow/tfjs-node-gpu';
import { AttentionModel } from '../src/assets/model/model.js';

import * as http from 'http';
import { Server } from 'socket.io';

const TIMEOUT_BETWEEN_EPOCHS_MS = 500;
const PORT = 8001;

// util function to sleep for a given ms
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Main function to start server, perform model training, and emit stats via the socket connection
async function run() {
  const port = process.env.PORT || PORT;
  const server = http.createServer();
  const io = new Server(server);

  let feat = 6;
  const data = tf.randomNormal([20, feat], 0, 1, 'float32', 64);
  console.log(data.shape);
  const ds = tf.data.array(data.clone().arraySync()).repeat(10).batch(10);

  server.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`);
  });

  io.on('connection', (socket) => {
    socket.on('startTraining', async (epochs) => {
      console.log("epochs recieved". epochs);
      // io.emit('predictResult', await pitch_type.predictSample(sample));
    });
  });
  
  let model = new AttentionModel({
    name: 'testing',
    feat:feat,
    ndim:64,
    heads:5,
    bijectLayers:5,
    encLayers:4,
    decLayers:4,
    seed:64,
    temp:0.01,
  });
  
  model.train(10, ds);

  io.emit('trainingComplete', true);
}

run();

