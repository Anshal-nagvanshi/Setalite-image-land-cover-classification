import mongoose from 'mongoose';

const SettingSchema = new mongoose.Schema({
  userId: {
    type: String,
    default: 'default_user'
  },
  architecture: {
    type: String,
    default: 'ResNet-50 (Pretrained on ImageNet / EuroSAT)'
  },
  normalization: {
    type: String,
    default: 'Min-Max Scaling [0, 1]'
  },
  confidenceThreshold: {
    type: Number,
    default: 15
  },
  spectralBands: {
    type: String,
    default: 'RGB (Red-Green-Blue Optical)'
  },
  theme: {
    type: String,
    default: 'Dark Emerald'
  },
  profile: {
    fullName: { type: String, default: '' },
    username: { type: String, default: '' },
    email: { type: String, default: '' }
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.models.Setting || mongoose.model('Setting', SettingSchema);
