FROM node:18-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM python:3.9-slim
ENV PORT=7860

WORKDIR /app

COPY --from=build /app/dist .

EXPOSE 7860

CMD ["sh", "-c", "python -m http.server ${PORT} --directory /app"]

