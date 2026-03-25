/*
MIT License

Copyright(c) 2022 Lukas Lipp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <imgui.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace ImOGuizmo {
	namespace internal {
		static struct Config {
			float mX = 0.f;
			float mY = 0.f;
			float mSize = 100.f;
			ImDrawList* mDrawList = nullptr;
		} config;

		struct ImVec3
		{
			ImVec3(const float x, const float y, const float z) : mData{ x, y, z } {}
			explicit ImVec3(const float* const data) : mData{ data[0], data[1], data[2] } {}
			float operator[](const int idx) const { return mData[idx]; }
			ImVec3 operator+(const ImVec3& other) const { return { mData[0] + other[0], mData[1] + other[1], mData[2] + other[2] }; }
			ImVec3 operator-(const ImVec3& other) const { return { mData[0] - other[0], mData[1] - other[1], mData[2] - other[2] }; }
			ImVec3 operator*(const float scalar) const { return { mData[0] * scalar, mData[1] * scalar, mData[2] * scalar }; }
			ImVec3 operator*(const ImVec3& other) const { return { mData[0] * other[0], mData[1] * other[1], mData[2] * other[2] }; }
			float mData[3];
		};
		inline ImVec4 multiply(const float* const m, const ImVec4& v)
		{
			const float x = m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w;
			const float y = m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w;
			const float z = m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w;
			const float w = m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w;
			return { x, y, z, w };
		}

		inline void multiply(const float* const l, const float* const r, float* out)
		{
			out[0] = l[0] * r[0] + l[1] * r[4] + l[2] * r[8] + l[3] * r[12];
			out[1] = l[0] * r[1] + l[1] * r[5] + l[2] * r[9] + l[3] * r[13];
			out[2] = l[0] * r[2] + l[1] * r[6] + l[2] * r[10] + l[3] * r[14];
			out[3] = l[0] * r[3] + l[1] * r[7] + l[2] * r[11] + l[3] * r[15];

			out[4] = l[4] * r[0] + l[5] * r[4] + l[6] * r[8] + l[7] * r[12];
			out[5] = l[4] * r[1] + l[5] * r[5] + l[6] * r[9] + l[7] * r[13];
			out[6] = l[4] * r[2] + l[5] * r[6] + l[6] * r[10] + l[7] * r[14];
			out[7] = l[4] * r[3] + l[5] * r[7] + l[6] * r[11] + l[7] * r[15];

			out[8] = l[8] * r[0] + l[9] * r[4] + l[10] * r[8] + l[11] * r[12];
			out[9] = l[8] * r[1] + l[9] * r[5] + l[10] * r[9] + l[11] * r[13];
			out[10] = l[8] * r[2] + l[9] * r[6] + l[10] * r[10] + l[11] * r[14];
			out[11] = l[8] * r[3] + l[9] * r[7] + l[10] * r[11] + l[11] * r[15];

			out[12] = l[12] * r[0] + l[13] * r[4] + l[14] * r[8] + l[15] * r[12];
			out[13] = l[12] * r[1] + l[13] * r[5] + l[14] * r[9] + l[15] * r[13];
			out[14] = l[12] * r[2] + l[13] * r[6] + l[14] * r[10] + l[15] * r[14];
			out[15] = l[12] * r[3] + l[13] * r[7] + l[14] * r[11] + l[15] * r[15];
		}

		inline float dot(const ImVec3& a, const ImVec3& b)
		{
			return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		}

		inline ImVec3 cross(const ImVec3& a, const ImVec3& b)
		{
			return {
			    a[1] * b[2] - a[2] * b[1],
				a[2] * b[0] - a[0] * b[2],
				a[0] * b[1] - a[1] * b[0]
			};
		}

		inline ImVec3 normalize(const ImVec3& a)
		{
			float il = 1.f / (sqrtf(dot(a, a)) + FLT_EPSILON);
			return a * il;
		}

		inline bool checkInsideCircle(const ImVec2 center, const float radius, const ImVec2 point)
		{
			return (point.x - center.x) * (point.x - center.x) + (point.y - center.y) * (point.y - center.y) <= radius * radius;
		}

		inline void drawPositiveLine(const ImVec2 center, const ImVec2 axis, const ImU32 color, const float radius, const float thickness, const char* text, const bool selected) {
			const auto lineEndPositive = ImVec2{ center.x + axis.x, center.y + axis.y };
			internal::config.mDrawList->AddLine(center, lineEndPositive, color, thickness);
			internal::config.mDrawList->AddCircleFilled(lineEndPositive, radius, color);
			const auto labelSize = ImGui::CalcTextSize(text);
			const auto textPos = ImVec2(floor(lineEndPositive.x - 0.5f * labelSize.x), floor(lineEndPositive.y - 0.5f * labelSize.y));
			if (selected) {
				internal::config.mDrawList->AddCircle(lineEndPositive, radius, IM_COL32_WHITE, 0, 1.1f);
				internal::config.mDrawList->AddText(textPos, IM_COL32_WHITE, text);
			}
			else internal::config.mDrawList->AddText(textPos, IM_COL32_BLACK, text);
		}

		inline void drawNegativeLine(const ImVec2 center, const ImVec2 axis, const ImU32 color, const float radius, const bool selected) {
			const auto lineEndNegative = ImVec2{ center.x - axis.x, center.y - axis.y };
			internal::config.mDrawList->AddCircleFilled(lineEndNegative, radius, color);
			if (selected) {
				internal::config.mDrawList->AddCircle(lineEndNegative, radius, IM_COL32_WHITE, 0, 1.1f);
			}
		}

		void lookAt(ImVec3 const& eye, ImVec3 const& at, ImVec3 const& up, float* viewMatrix)
		{
#ifdef IMOGUIZMO_LEFT_HANDED
			const auto f = normalize(at - eye);
			const auto r = normalize(cross(up, f));
			const auto u = cross(f, r);
			viewMatrix[0] = r[0]; viewMatrix[1] = u[0]; viewMatrix[2] = f[0]; viewMatrix[3] = 0.0f;
			viewMatrix[4] = r[1]; viewMatrix[5] = u[1]; viewMatrix[6] = f[1]; viewMatrix[7] = 0.0f;
			viewMatrix[8] = r[2]; viewMatrix[9] = u[2]; viewMatrix[10] = f[2]; viewMatrix[11] = 0.0f;
			viewMatrix[12] = -dot(r, eye); viewMatrix[13] = -dot(u, eye); viewMatrix[14] = -dot(f, eye); viewMatrix[15] = 1.0f;
#else
			const auto f = normalize(at - eye);
			const auto r = normalize(cross(f, up));
			const auto u = cross(r, f);
			viewMatrix[0] = r[0]; viewMatrix[1] = u[0]; viewMatrix[2] = -f[0]; viewMatrix[3] = 0.0f;
			viewMatrix[4] = r[1]; viewMatrix[5] = u[1]; viewMatrix[6] = -f[1]; viewMatrix[7] = 0.0f;
			viewMatrix[8] = r[2]; viewMatrix[9] = u[2]; viewMatrix[10] = -f[2]; viewMatrix[11] = 0.0f;
			viewMatrix[12] = -dot(r, eye); viewMatrix[13] = -dot(u, eye); viewMatrix[14] = dot(f, eye); viewMatrix[15] = 1.0f;
#endif
		}

		inline void invert4x4(const float* m, float* out)
		{
			out[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
			out[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
			out[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
			out[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
			out[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
			out[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
			out[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
			out[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
			out[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
			out[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
			out[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
			out[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
			out[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
			out[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
			out[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
			out[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

			float det = m[0] * out[0] + m[1] * out[4] + m[2] * out[8] + m[3] * out[12];
			det = 1.0f / det;
			for (unsigned int i = 0; i < 16; i++) out[i] = out[i] * det;
		}
	}

	static struct Config {
		// in relation to half the rect size
		float lineThicknessScale = 0.017f;
		float axisLengthScale = 0.33f;
		float positiveRadiusScale = 0.075f;
		float negativeRadiusScale = 0.05f;
		float hoverCircleRadiusScale = 0.88f;
		ImU32 xCircleFrontColor = IM_COL32(255, 54, 83, 255);
		ImU32 xCircleBackColor = IM_COL32(154, 57, 71, 255);
		ImU32 yCircleFrontColor = IM_COL32(138, 219, 0, 255);
		ImU32 yCircleBackColor = IM_COL32(98, 138, 34, 255);
		ImU32 zCircleFrontColor = IM_COL32(44, 143, 255, 255);
		ImU32 zCircleBackColor = IM_COL32(52, 100, 154, 255);
		ImU32 hoverCircleColor = IM_COL32(100, 100, 100, 130);
	} config;

	inline void SetRect(const float x, const float y, const float size)
	{
		internal::config.mX = x;
		internal::config.mY = y;
		internal::config.mSize = size;
	}

	inline void SetDrawList(ImDrawList* drawlist = nullptr)
	{
		internal::config.mDrawList = drawlist ? drawlist : ImGui::GetWindowDrawList();
	}

	inline void BeginFrame(const bool background = false)
	{
		const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus | ((background != true) ? ImGuiWindowFlags_NoBackground : ImGuiWindowFlags_None);
		ImGui::SetNextWindowPos({ internal::config.mX, internal::config.mY }, ImGuiCond_Always);
		ImGui::SetNextWindowSize({ internal::config.mSize, internal::config.mSize });
		ImGui::Begin("imoguizmo", nullptr, flags);
		SetDrawList(internal::config.mDrawList);
		ImGui::End();
	}

	inline bool DrawGizmo(float* const viewMatrix, const float* const projectionMatrix, const float pivotDistance = 0.0f) {
		const float size = internal::config.mSize;
		const float hSize = size * 0.5f;
		const auto center = ImVec2{ internal::config.mX + hSize, internal::config.mY + hSize };

		float viewProjection[16];
		internal::multiply(viewMatrix, projectionMatrix, viewProjection);
		// correction for non-square aspect ratio
		{
			const float aspectRatio = projectionMatrix[5] / projectionMatrix[0];
			viewProjection[0] *= aspectRatio; viewProjection[8] *= aspectRatio;
		}
		// axis
		const float axisLength = size * config.axisLengthScale;
		const ImVec4 xAxis = internal::multiply(viewProjection, ImVec4{ axisLength, 0, 0, 0 });
		const ImVec4 yAxis = internal::multiply(viewProjection, ImVec4{ 0, axisLength, 0, 0 });
		const ImVec4 zAxis = internal::multiply(viewProjection, ImVec4{ 0, 0, axisLength, 0 });

		const bool interactive = pivotDistance > 0.0f;
		const ImVec2 mousePos = ImGui::GetIO().MousePos;

		const float hoverCircleRadius = hSize * config.hoverCircleRadiusScale;
		SetDrawList(internal::config.mDrawList);
		if (config.hoverCircleColor != 0 && interactive && internal::checkInsideCircle(center, hoverCircleRadius, mousePos)) internal::config.mDrawList->AddCircleFilled(center, hoverCircleRadius, config.hoverCircleColor);

		const float positiveRadius = size * config.positiveRadiusScale;
		const float negativeRadius = size * config.negativeRadiusScale;
		const bool xPositiveCloser = 0.0f >= xAxis.w;
		const bool yPositiveCloser = 0.0f >= yAxis.w;
		const bool zPositiveCloser = 0.0f >= zAxis.w;

		// sort axis based on distance
		// 0 : +x axis, 1 : +y axis, 2 : +z axis, 3 : -x axis, 4 : -y axis, 5 : -z axis
		std::vector<std::pair<int, float>> pairs = { {0, xAxis.w}, {1, yAxis.w}, {2, zAxis.w}, {3, -xAxis.w}, {4, -yAxis.w}, {5, -zAxis.w} };
		sort(pairs.begin(), pairs.end(), [=](const std::pair<int, float>& aA, const std::pair<int, float>& aB) { return aA.second > aB.second; });

		// find selection, front to back
		int selection = -1;
		for (auto it = pairs.crbegin(); it != pairs.crend() && selection == -1 && interactive; ++it) {
			switch (it->first) {
			case 0: // +x axis
				if (internal::checkInsideCircle(ImVec2{ center.x + xAxis.x, center.y - xAxis.y }, positiveRadius, mousePos)) selection = 0;
				break;
			case 1: // +y axis
				if (internal::checkInsideCircle(ImVec2{ center.x + yAxis.x, center.y - yAxis.y }, positiveRadius, mousePos)) selection = 1;
				break;
			case 2: // +z axis
				if (internal::checkInsideCircle(ImVec2{ center.x + zAxis.x, center.y - zAxis.y }, positiveRadius, mousePos)) selection = 2;
				break;
			case 3: // -x axis
				if (internal::checkInsideCircle(ImVec2{ center.x - xAxis.x, center.y + xAxis.y }, negativeRadius, mousePos)) selection = 3;
				break;
			case 4: // -y axis
				if (internal::checkInsideCircle(ImVec2{ center.x - yAxis.x, center.y + yAxis.y }, negativeRadius, mousePos)) selection = 4;
				break;
			case 5: // -z axis
				if (internal::checkInsideCircle(ImVec2{ center.x - zAxis.x, center.y + zAxis.y }, negativeRadius, mousePos)) selection = 5;
				break;
			default: break;
			}
		}

		// draw back first
		const float lineThickness = size * config.lineThicknessScale;
		for (const auto& [fst, snd] : pairs) {
			switch (fst) {
			case 0: // +x axis
				internal::drawPositiveLine(center, ImVec2{ xAxis.x, -xAxis.y }, xPositiveCloser ? config.xCircleFrontColor : config.xCircleBackColor, positiveRadius, lineThickness, "X", selection == 0);
				continue;
			case 1: // +y axis
				internal::drawPositiveLine(center, ImVec2{ yAxis.x, -yAxis.y }, yPositiveCloser ? config.yCircleFrontColor : config.yCircleBackColor, positiveRadius, lineThickness, "Y", selection == 1);
				continue;
			case 2: // +z axis
				internal::drawPositiveLine(center, ImVec2{ zAxis.x, -zAxis.y }, zPositiveCloser ? config.zCircleFrontColor : config.zCircleBackColor, positiveRadius, lineThickness, "Z", selection == 2);
				continue;
			case 3: // -x axis
				internal::drawNegativeLine(center, ImVec2{ xAxis.x, -xAxis.y }, !xPositiveCloser ? config.xCircleFrontColor : config.xCircleBackColor, negativeRadius, selection == 3);
				continue;
			case 4: // -y axis
				internal::drawNegativeLine(center, ImVec2{ yAxis.x, -yAxis.y }, !yPositiveCloser ? config.yCircleFrontColor : config.yCircleBackColor, negativeRadius, selection == 4);
				continue;
			case 5: // -z axis
				internal::drawNegativeLine(center, ImVec2{ zAxis.x, -zAxis.y }, !zPositiveCloser ? config.zCircleFrontColor : config.zCircleBackColor, negativeRadius, selection == 5);
				continue;
			default: break;
			}
		}
		internal::config.mDrawList = nullptr;

		if (selection != -1 && ImGui::IsMouseClicked(ImGuiPopupFlags_MouseButtonLeft)) {
			float modelMat[16];
			internal::invert4x4(viewMatrix, modelMat);

			const internal::ImVec3 pivotPos = internal::ImVec3{ &modelMat[12] } - internal::ImVec3{ &modelMat[8] } *pivotDistance;

			internal::ImVec3 ups[6] = { 
#ifdef IMOGUIZMO_LEFT_HANDED
#ifdef IMOGUIZMO_Z_UP
				{0, 0, 1}, {0, 0, 1}, {0, -1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 1, 0}
#else // Y is up
				{0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 1, 0}
#endif
#else // IMOGUIZMO_RIGHT_HANDED
#ifdef IMOGUIZMO_Z_UP
				{0, 0, 1}, {0, 0, 1}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, -1, 0}
#else // Y is up
				{0, 1, 0}, {0, 0, -1}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0} 
#endif
#endif
			};
			// +x axis
			if (selection == 0) internal::lookAt(pivotPos + internal::ImVec3{ pivotDistance, 0, 0 }, pivotPos, ups[0], viewMatrix);
			// +y axis
			if (selection == 1) internal::lookAt(pivotPos + internal::ImVec3{ 0, pivotDistance, 0 }, pivotPos, ups[1], viewMatrix);
			// +z axis
			if (selection == 2) internal::lookAt(pivotPos + internal::ImVec3{ 0, 0, pivotDistance }, pivotPos, ups[2], viewMatrix);
			// -x axis 
			if (selection == 3) internal::lookAt(pivotPos - internal::ImVec3{ pivotDistance, 0, 0 }, pivotPos, ups[3], viewMatrix);
			// -y axis 
			if (selection == 4) internal::lookAt(pivotPos - internal::ImVec3{ 0, pivotDistance, 0 }, pivotPos, ups[4], viewMatrix);
			// -z axis 
			if (selection == 5) internal::lookAt(pivotPos - internal::ImVec3{ 0, 0, pivotDistance }, pivotPos, ups[5], viewMatrix);

			return true;
		}
		
		return false;
	}
}
