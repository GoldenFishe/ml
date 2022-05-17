const fs = require("fs/promises");
const tf = require("@tensorflow/tfjs");

const BATCH_SIZE = 32;

async function getFilenames(type, dataType) {
    try {
        return fs.readdir(`${__dirname}/data/${dataType}/${type}`);
    } catch (err) {
        console.log(err);
    }
}

async function getFile(type, filename, dataType) {
    try {
        const file = await fs.readFile(`${__dirname}/data/${dataType}/${type}/${filename}`);
        const pixelData = {width: 300, height: 300, data: file};
        return tf.browser.fromPixels(pixelData);
    } catch (err) {
        console.error(err);
    }
}

async function getData(type, dataType) {
    const filenames = await getFilenames(type, dataType);
    const data = [];
    for (let filename of filenames) {
        let file = await getFile(type, filename, dataType);
        file = file.div(255);
        data.push(file);
    }
    return tf.data.array(data);
}

async function getLabels(size, label) {
    const labels = [];
    for (let i = 0; i < size; i++) {
        labels.push(tf.tensor(label));
    }
    return tf.data.array(labels);
}

async function getDataset(dataType) {
    const paperData = await getData('paper', dataType);
    const rockData = await getData('rock', dataType);
    const scissorsData = await getData('scissors', dataType);

    const paperLabels = await getLabels(paperData.size, [1, 0, 0]);
    const rockLabels = await getLabels(rockData.size, [0, 1, 0]);
    const scissorsLabels = await getLabels(scissorsData.size, [0, 0, 1]);

    const data = paperData.concatenate(rockData).concatenate(scissorsData);
    const labels = paperLabels.concatenate(rockLabels).concatenate(scissorsLabels);

    return tf.data.zip({xs: data, ys: labels}).shuffle(372, Math.random().toString()).batch(8);
}

async function getTrainDataset() {
    return getDataset('train');
}

async function getTestDataset() {
    return getDataset('test');
}

module.exports = {getTrainDataset, getTestDataset};