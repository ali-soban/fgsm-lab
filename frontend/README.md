This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## FGSM

FGSM is basically a way of fooling a NN. As we know, models rely on gradient changes to minimize losses, FGSM basically checks the gradient and maximises loss (Gradient Ascent), to make the model less sure about its prediction and confuse it. This causes the model's confidence scores to go down and it can make wrong predictions.

Now, regarding the epsilon. It can be seen as a multiplier of the effort put in by the attack. The higher the epsilon, the bugger changes made by the attack hence more chances of a wrong prediction.
