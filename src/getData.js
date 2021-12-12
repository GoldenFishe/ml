const fs = require("fs/promises");
const tf = require("@tensorflow/tfjs");

const SHUFFLE = 10;

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
        const file = await getFile(type, filename, dataType);
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

async function getTrainDataset() {
    const paperData = await getData('paper', 'train');
    const rockData = await getData('rock', 'train');
    const scissorsData = await getData('scissors', 'train');

    const paperLabels = await getLabels(paperData.size, 1);
    const rockLabels = await getLabels(rockData.size, 2);
    const scissorsLabels = await getLabels(scissorsData.size, 3);

    const data = paperData.concatenate(rockData).concatenate(scissorsData);
    const labels = paperLabels.concatenate(rockLabels).concatenate(scissorsLabels);

    return tf.data.zip({xs: data, ys: labels}).batch(3);
}

async function getTestDataset() {
    const paperData = await getData('paper', 'test');
    const rockData = await getData('rock', 'test');
    const scissorsData = await getData('scissors', 'test');

    const paperLabels = await getLabels(paperData.size, 1);
    const rockLabels = await getLabels(rockData.size, 2);
    const scissorsLabels = await getLabels(scissorsData.size, 3);

    const data = paperData.concatenate(rockData).concatenate(scissorsData);
    const labels = paperLabels.concatenate(rockLabels).concatenate(scissorsLabels);

    return tf.data.zip({xs: data, ys: labels}).batch(3).shuffle(SHUFFLE);
}

module.exports = {
    getTrainDataset,
    getTestDataset
};