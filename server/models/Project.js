import mongoose from 'mongoose';

const ProjectSchema = new mongoose.Schema({
  userId: {
    type: String,
    index: true
  },
  name: {
    type: String,
    required: true,
    trim: true
  },
  description: {
    type: String,
    default: 'Multi-spectral satellite land categorization batch.'
  },
  model: {
    type: String,
    default: 'ResNet-50'
  },
  status: {
    type: String,
    enum: ['Active', 'Archived', 'Pending'],
    default: 'Active'
  },
  accuracy: {
    type: String,
    default: '98.7% Acc'
  },
  images: [
    {
      _id: { type: String },
      name: { type: String, default: 'satellite_patch.jpg' },
      imageUrl: { type: String },
      topPrediction: {
        className: { type: String },
        confidence: { type: Number }
      },
      probabilities: [
        {
          label: { type: String },
          confidence: { type: Number }
        }
      ],
      createdAt: { type: Date, default: Date.now }
    }
  ],
  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.models.Project || mongoose.model('Project', ProjectSchema);
