constrctData = constrctDataFunction;
readData = readDataFunction;
generateModel = generateModelFunction;
trainModel = trainModelFunction;
showData = showDataFunction;
assesModel = assesModelFunction;

async function runModel() {
  // asnc calls
  const data = await readData();
  showData(data); //display the data

  const model = generateModel();
  tfvis.show.modelSummary({ name: "Summary of the Model" }, model); //show model summary

  const tensorData = constrctData(data); //prepare data
  const { inputs, outputs } = tensorData;

  await trainModel(model, inputs, outputs, 10); //train data
  console.log("Training is Completed"); //showing progress

  await assesModel(model, inputs, outputs);
}

async function readDataFunction() {
  const covidItalyDataReq = await fetch("https://raw.githubusercontent.com/furkanacundas/COVAID_20/master/covid19-italy-region-beds.json"); //asynchronous function
  const covidItalyData = await covidItalyDataReq.json();
  return covidItalyData;
}

function singlePlot(values, name, xoutput, youtput) {
  tfvis.render.scatterplot(
    { name: name },
    { values },
    {
      xoutput: xoutput,
      youtput: youtput,
      height: 300,
    }
  );
}
function showDataFunction(data) {
  let showData = data.map((d) => ({
    x: d.totalCases, // total no of covid cases on x-axis
    y: d.intensive_care, // total no intensive care patients on y-axis
  }));

  singlePlot(
    showData,
    "Total Cases vs Intensive Care",
    "totalCases",
    "intensive_care"
  );
}
function generateModelFunction() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [2],
      units: 10,
      useBias: true,
      activation: "relu",
    })
  );
  model.add(tf.layers.dense({ units: 50, useBias: true, activation: "tanh" }));
  model.add(tf.layers.dense({ units: 25, useBias: true, activation: "relu" }));
  model.add(
    tf.layers.dense({ units: 10, useBias: true, activation: "softmax" })
  );

  return model;
}
function extractInputs(data) {
  let inputs = [];
  inputs = data.map((d) => [d.totalCases, d.intensive_care]);
  return inputs;
}
function constrctDataFunction(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = extractInputs(data);
    const outputs = data.map((d) => d.quality);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, inputs[0].length]);
    const outputTensor = tf.oneHot(tf.tensor1d(outputs, "int32"), 10);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const outputMax = outputTensor.max();
    const outputMin = outputTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedoutputs = outputTensor
      .sub(outputMin)
      .div(outputMax.sub(outputMin));

    return {
      inputs: normalizedInputs,
      outputs: normalizedoutputs,
      inputMax,
      inputMin,
      outputMax,
      outputMin,
    };
  });
}
async function trainModelFunction(model, inputs, outputs, epochs) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const batchSize = 128;

  return await model.fit(inputs, outputs, {
    batchSize,
    epochs,
    shuffle: false,
    callbacks: tfvis.show.fitCallbacks(
      { name: "The Performance of the Taining Dataset" },
      ["loss", "accuracy"],
      { height: 300, callbacks: ["onEpochEnd"] }
    ),
  });
}
async function assesModelFunction(model, inputs, outputs) {
  const result = await model.evaluate(inputs, outputs, { batchSize: 128 });
  console.log("The Accuracy of the Model is:");
  result[1].print();
}

document.addEventListener("DOMContentLoaded", runModel); // display data object model on browser
