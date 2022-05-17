const tf = require("@tensorflow/tfjs-node")

const {getTrainDataset, getTestDataset} = require("./getData");

const IMAGE_WIDTH = 300;
const IMAGE_HEIGHT = 300;
const IMAGE_CHANNELS = 3;
const NUM_OUTPUT_CLASSES = 3;
const CLASSES = ["paper", "rock", "scissors"];

function getModel() {
    const model = tf.sequential();
    const inputLayer = tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 10,
        filters: 8,
        strides: 1,
        activation: "relu"
    });
    const hiddenLayer1 = tf.layers.maxPool2d({poolSize: [2, 2], strides: [2, 2]});
    const hiddenLayer2 = tf.layers.conv2d({
        kernelSize: 10,
        filters: 16,
        strides: 1,
        activation: "relu"
    });
    const hiddenLayer3 = tf.layers.maxPool2d({poolSize: [2, 2], strides: [2, 2]});
    const hiddenLayer4 = tf.layers.flatten();
    const outputLayer = tf.layers.dense({units: NUM_OUTPUT_CLASSES, activation: "softmax"});

    model.add(inputLayer);
    model.add(hiddenLayer1);
    model.add(hiddenLayer2);
    model.add(hiddenLayer3);
    model.add(hiddenLayer4);
    model.add(outputLayer);

    const optimizer = tf.train.adam(0.0001);

    model.compile({
        optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}

async function train(model) {
    const [trainDataset, testDataset] = await Promise.all([getTrainDataset(), getTestDataset()]);
    console.log("start");
    const {history} = await model.fitDataset(trainDataset, {
        epochs: 15,
        validationData: testDataset,
        validationBatches: 1,
        validationBatchSize: 8,
        callbacks: {
            onEpochEnd: (epoch, logs) => console.log(logs.loss)
        }
    })
    console.log(history);
    console.log("end");
}

async function predict(model) {
    const testDataset = await getTestDataset();
    const results = {};

    await testDataset.take(1).forEachAsync(async ({xs, ys}) => {
        const predict = model.predict(xs);
        const [target, res] = await Promise.all([ys.array(), predict.array()])
        results.target = target;
        results.res = res;
    });
    console.log({target: results.target.slice(0, 5)});
    console.log({res: results.res.slice(0, 5)});
}

async function main() {
    const model = getModel();
    await train(model);
    await predict(model);
}

main();