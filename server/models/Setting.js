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
  dataAugmentation: {
    type: Boolean,
    default: true
  },
  profile: {
    fullName: { type: String, default: 'Anshal Nagyanshi' },
    username: { type: String, default: 'anshal_nagyanshi' },
    email: { type: String, default: 'anshal@example.com' }
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.models.Setting || mongoose.model('Setting', SettingSchema);
