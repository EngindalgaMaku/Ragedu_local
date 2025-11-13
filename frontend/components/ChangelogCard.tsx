"use client";
import React from "react";

const InfoIcon = () => (
  <svg
    className="w-6 h-6"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
    ></path>
  </svg>
);

const GoalIcon = () => (
  <svg
    className="w-6 h-6"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-.806 1.946 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"
    ></path>
  </svg>
);

const UniversityIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
    />
  </svg>
);

const ChangelogCard = () => {
  return (
    <div className="bg-gradient-to-br from-white via-blue-50/50 to-indigo-50/30 p-8 rounded-2xl shadow-xl border border-blue-100/60 h-full backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center gap-3 mb-8">
        <div className="p-3 bg-gradient-to-br from-blue-500 via-indigo-600 to-purple-600 text-white rounded-xl shadow-lg">
          <InfoIcon />
        </div>
        <div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Proje Bilgileri
          </h2>
          <p className="text-gray-600 text-sm mt-0.5">Geliştirme ve Amaç</p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Proje Bilgileri Section */}
        <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-blue-100/50 shadow-sm">
          <div className="flex items-start gap-4 mb-4">
            <div className="p-2.5 bg-gradient-to-br from-blue-100 to-indigo-100 text-blue-600 rounded-lg flex-shrink-0">
              <UniversityIcon />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-gray-900 mb-3 text-lg">Geliştirme Bilgileri</h3>
              <div className="space-y-2.5 text-sm leading-relaxed text-gray-700">
                <p>
                  <span className="font-semibold text-gray-900">Üniversite:</span>{" "}
                  <span className="text-gray-700">Burdur Mehmet Akif Ersoy Üniversitesi</span>
                </p>
                <p>
                  <span className="font-semibold text-gray-900">Bölüm:</span>{" "}
                  <span className="text-gray-700">Yazılım Mühendisliği</span>
                </p>
                <p>
                  <span className="font-semibold text-gray-900">Program:</span>{" "}
                  <span className="text-gray-700">Yüksek Lisans</span>
                </p>
                <div className="pt-2 border-t border-gray-200 mt-3">
                  <p className="mb-2">
                    <span className="font-semibold text-gray-900">Geliştirici:</span>{" "}
                    <span className="text-blue-700 font-medium">Engin DALGA</span>
                  </p>
                  <p>
                    <span className="font-semibold text-gray-900">Danışman:</span>{" "}
                    <span className="text-indigo-700 font-medium">Prof. Dr. Serkan BALLI</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-blue-200/60 my-6"></div>

        {/* Proje Amacı Section */}
        <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-blue-100/50 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="p-2.5 bg-gradient-to-br from-indigo-100 to-purple-100 text-indigo-600 rounded-lg flex-shrink-0">
              <GoalIcon />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-gray-900 mb-3 text-lg">Proje Amacı</h3>
              <div className="space-y-3 text-sm leading-relaxed text-gray-700">
                <p>
                  Bu proje, <span className="font-semibold text-gray-900">RAG (Retrieval Augmented Generation)</span> temelli chatbot teknolojisinin eğitim alanındaki uygulamalarını araştırmayı hedeflemektedir.
                </p>
                <div className="bg-blue-50/50 rounded-lg p-4 border-l-4 border-blue-400">
                  <p className="font-medium text-gray-900 mb-2">Ana Hedefler:</p>
                  <ul className="space-y-1.5 text-gray-700 list-disc list-inside">
                    <li>Kişiselleştirilmiş eğitim ortamlarına katkı sağlamak</li>
                    <li>Öğretmen-öğrenci etkileşimini yapay zeka ile desteklemek</li>
                    <li>Eğitim kalitesini artırmak ve öğrenme deneyimini iyileştirmek</li>
                    <li>RAG teknolojisinin eğitimdeki potansiyelini keşfetmek</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 border-t border-gray-200/60">
          <p className="text-xs text-gray-500 text-center">
            © 2025 Engin DALGA - MAKÜ Yüksek Lisans Ödevi
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChangelogCard;
