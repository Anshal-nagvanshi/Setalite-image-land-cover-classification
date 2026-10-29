import mongoose from 'mongoose';

const ClassificationSchema = new mongoose.Schema({
  title: {
    type: String,
    default: 'Satellite Patch Analysis'
  },
  patchId: {
    type: String,
    default: () => 'LCC_' + Math.floor(10000 + Math.random() * 90000)
  },
  coordinates: {
    type: String,
    default: '28°N, 77°E'
  },
  imageUrl: {
    type: String,
    required: true
  },
  modelUsed: {
    type: String,
    default: 'ResNet-50'
  },
  topPrediction: {
    className: { type: String, required: true },
    confidence: { type: Number, required: true }
  },
  probabilities: [
    {
      label: { type: String, required: true },
      confidence: { type: Number, required: true }
    }
  ],
  projectId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Project',
    default: null
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.models.Classification || mongoose.model('Classification', ClassificationSchema);
