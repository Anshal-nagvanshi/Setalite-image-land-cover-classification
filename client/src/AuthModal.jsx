import React, { useState } from 'react';

export default function AuthModal({ isOpen, onClose, onAuthSuccess, showToast }) {
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({
    fullName: '',
    username: '',
    email: '',
    password: '',
    identifier: ''
  });
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  if (!isOpen) return null;

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setErrorMessage('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMessage('');

    const endpoint = isSignUp ? '/api/auth/register' : '/api/auth/login';
    const payload = isSignUp
      ? {
          fullName: formData.fullName,
          username: formData.username,
          email: formData.email,
          password: formData.password
        }
      : {
          identifier: formData.identifier,
          password: formData.password
        };

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Authentication failed');
      }

      // Success
      localStorage.setItem('auth_token', data.token);
      localStorage.setItem('auth_user', JSON.stringify(data.user));
      onAuthSuccess(data.user);
      showToast(isSignUp ? 'Account created successfully! Welcome.' : `Welcome back, ${data.user.fullName}!`);
      onClose();
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-[#090d16]/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-[#0f172a] border border-slate-800 p-8 rounded-3xl w-full max-w-md shadow-2xl relative">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 text-slate-400 hover:text-white transition p-1"
        >
          <i className="fa-solid fa-xmark text-base"></i>
        </button>

        {/* Brand Icon & Heading */}
        <div className="text-center mb-6">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto mb-3 text-xl">
            <i className="fa-solid fa-earth-americas"></i>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            {isSignUp ? 'Create Researcher Account' : 'Sign In to Satellite LCC'}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            {isSignUp
              ? 'Join to save classifications and project batches'
              : 'Enter your credentials to access your workspaces'}
          </p>
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2">
            <i className="fa-solid fa-circle-exclamation"></i>
            <span>{errorMessage}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          {isSignUp ? (
            <>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Full Name</label>
                <input
                  type="text"
                  name="fullName"
                  required
                  value={formData.fullName}
                  onChange={handleChange}
                  placeholder="e.g. Anshal Nagyanshi"
                  className="w-full bg-[#090d16] border border-slate-700/80 rounded-xl p-3 text-slate-200 focus:border-emerald-500 outline-none transition"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Username</label>
                <input
                  type="text"
                  name="username"
                  required
                  value={formData.username}
                  onChange={handleChange}
                  placeholder="e.g. anshal_geo"
                  className="w-full bg-[#090d16] border border-slate-700/80 rounded-xl p-3 text-slate-200 focus:border-emerald-500 outline-none transition"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Email Address</label>
                <input
                  type="email"
                  name="email"
                  required
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="researcher@satellite.org"
                  className="w-full bg-[#090d16] border border-slate-700/80 rounded-xl p-3 text-slate-200 focus:border-emerald-500 outline-none transition"
                />
              </div>

              <div>
                <label className="block text-slate-300 font-medium mb-1">Password</label>
                <input
                  type="password"
                  name="password"
                  required
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="••••••••"
                  className="w-full bg-[#090d16] border border-slate-700/80 rounded-xl p-3 text-slate-200 focus:border-emerald-500 outline-none transition"
                />
              </div>
            </>
          ) : (
            <>
              <div>
                <label className="block text-slate-300 font-medium mb-1">Email Address</label>
                <input
                  type="email"
                  name="identifier"
                  required
                  value={formData.identifier}
                  onChange={handleChange}
                  placeholder="name@example.com"
                  className="w-full bg-[#090d16] border border-slate-700/80 rounded-xl p-3 text-slate-200 focus:border-emerald-500 outline-none transition"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-slate-300 font-medium">Password</label>
                  <span className="text-[10px] text-emerald-400 cursor-pointer hover:underline">Forgot?</span>
                </div>
                <input
                  type="password"
                  name="password"
                  required
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="••••••••"
                  className="w-full bg-[#090d16] border border-slate-700/80 rounded-xl p-3 text-slate-200 focus:border-emerald-500 outline-none transition"
                />
              </div>
            </>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 text-white font-medium rounded-xl transition shadow-lg shadow-emerald-900/30 flex justify-center items-center gap-2 text-xs"
          >
            {loading ? (
              <i className="fa-solid fa-spinner animate-spin"></i>
            ) : (
              <>
                <i className={`fa-solid ${isSignUp ? 'fa-user-plus' : 'fa-right-to-bracket'}`}></i>
                <span>{isSignUp ? 'Create Account' : 'Sign In'}</span>
              </>
            )}
          </button>
        </form>

        {/* Toggle between Sign In and Sign Up */}
        <div className="mt-6 text-center text-xs text-slate-400 border-t border-slate-800 pt-4">
          {isSignUp ? (
            <p>
              Already have an account?{' '}
              <button
                type="button"
                onClick={() => { setIsSignUp(false); setErrorMessage(''); }}
                className="text-emerald-400 font-semibold hover:underline"
              >
                Sign In
              </button>
            </p>
          ) : (
            <p>
              Don't have an account yet?{' '}
              <button
                type="button"
                onClick={() => { setIsSignUp(true); setErrorMessage(''); }}
                className="text-emerald-400 font-semibold hover:underline"
              >
                Sign Up
              </button>
            </p>
          )}
        </div>

      </div>
    </div>
  );
}
